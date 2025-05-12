# Import RPB model along with other utility functions

from idaes.core import FlowsheetBlock
from idaes.models.unit_models import Feed, Product
from RPB_model import RotaryPackedBed

from pyomo.environ import (
    ConcreteModel,
    SolverFactory,
    TransformationFactory,
    Reference,
    units as pyunits,
    Param,
    Suffix,
)

import idaes.core.util as iutil
import idaes.core.util.scaling as iscale
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.logger as idaeslog
from idaes.core.util.initialization import propagate_state

from idaes.models_extra.power_generation.properties import FlueGasParameterBlock
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)
from idaes.models_extra.power_generation.properties.natural_gas_PR import (
    get_prop,
    EosType,
)

from pyomo.network import Arc

from pyomo.contrib.parmest.experiment import Experiment

from idaes.core.util.model_diagnostics import DiagnosticsToolbox

import numpy as np

class RPB_experiment(Experiment):
    def __init__(self, theta_initial=None, rpm=0.01, pressure=1.01e5):
        """
        Arguments
        ---------
        theta_initial: dictionary, initial guesses for the unknown parameters
        rpm: float, rotations per minute design spec
        pressure: float, pressure design spec for inlet gas, Pa
        
        """
        # Make the dictionary
        self.theta_initial = {}

        if theta_initial is None:
            # Default values from RPB_model
            self.theta_initial['hgx'] = 25 * 1e-3
            self.theta_initial['C1'] = 2.562434e-12
            self.theta_initial['delH_1'] = 98.76
            self.theta_initial['delH_2'] = 77.11
            self.theta_initial['delH_3'] = 21.25
        
        # Specified intial decision variable values
        self.w_rpm = rpm
        self.pressure = pressure

        self.model = None
    
    def get_labeled_model(self):
        if self.model is None:
            self.create_model()
            self.finalize_model()
            self.label_experiment()
        return self.model
    
    def create_model(self):
        """
        Method to create an unlabled model RPB system.
        
        """
        # create Flowsheet block
        m = self.model = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic = False)

        # create gas phase properties block
        flue_species={"H2O", "CO2", "N2"}
        prop_config = get_prop(flue_species, ["Vap"], eos=EosType.IDEAL)
        prop_config["state_bounds"]["pressure"] = (0.99*1e5,1.02*1e5,1.2*1e5, pyunits.Pa)
        prop_config["state_bounds"]["temperature"] = (25+273.15,90+273.15,180+273.15, pyunits.K)

        m.fs.gas_props = GenericParameterBlock(
            **prop_config,
            doc = "Flue gas properties",
        )

        m.fs.gas_props.set_default_scaling("temperature", 1e-2)
        m.fs.gas_props.set_default_scaling("pressure", 1e-4)
        
        # create feed and product blocks
        m.fs.flue_gas_in = Feed(property_package = m.fs.gas_props)
        m.fs.flue_gas_out = Product(property_package = m.fs.gas_props)
        m.fs.steam_sweep_feed = Feed(property_package = m.fs.gas_props)
        m.fs.regeneration_prod = Product(property_package = m.fs.gas_props)

        # limited discretization, much faster
        # m.fs.RPB = RotaryPackedBed(
            # property_package = m.fs.gas_props,
            # z_init_points = (0.01,0.99),
            # o_init_points = (0.01,0.99),
        # )
        
        # Larger discretization
        z_init_points=tuple(np.geomspace(0.01, 0.5, 9)[:-1]) + tuple((1 - np.geomspace(0.01, 0.5, 9))[::-1])
        o_init_points=tuple(np.geomspace(0.005, 0.1, 8)) + tuple(np.linspace(0.1, 0.995, 10)[1:])

        z_nfe=20
        o_nfe=20

        m.fs.RPB = RotaryPackedBed(
            property_package = m.fs.gas_props,
            z_init_points=z_init_points,
            o_init_points=o_init_points,
            z_nfe=z_nfe,
            o_nfe=o_nfe,
        )

        # add stream connections
        m.fs.s_flue_gas = Arc(source=m.fs.flue_gas_in.outlet, destination=m.fs.RPB.ads_gas_inlet)
        m.fs.s_cleaned_flue_gas = Arc(source=m.fs.RPB.ads_gas_outlet, destination=m.fs.flue_gas_out.inlet)
        m.fs.s_steam_feed = Arc(source=m.fs.steam_sweep_feed.outlet, destination=m.fs.RPB.des_gas_inlet)
        m.fs.s_regeneration_prod = Arc(source=m.fs.RPB.des_gas_outlet, destination=m.fs.regeneration_prod.inlet)

        TransformationFactory("network.expand_arcs").apply_to(m)

        # fix state variables in feed and product blocks
        # ads side
        m.fs.flue_gas_in.pressure.fix(1.02*1e5)
        m.fs.flue_gas_in.temperature.fix(90+273.15)
        m.fs.flue_gas_out.pressure.fix(1.01325*1e5)
        m.fs.flue_gas_in.mole_frac_comp[0,"CO2"].fix(0.04)
        m.fs.flue_gas_in.mole_frac_comp[0,"H2O"].fix(0.09)
        m.fs.flue_gas_in.mole_frac_comp[0,"N2"].fix(1-0.04-0.09)

        #des side
        m.fs.steam_sweep_feed.pressure.fix(1.05*1e5)
        m.fs.steam_sweep_feed.temperature.fix(120+273.15)
        m.fs.regeneration_prod.pressure.fix(1.01325*1e5)
        m.fs.steam_sweep_feed.mole_frac_comp[0,"CO2"].fix(1e-5)
        m.fs.steam_sweep_feed.mole_frac_comp[0,"N2"].fix(1e-3)
        m.fs.steam_sweep_feed.mole_frac_comp[0,"H2O"].fix(1-1e-5-1e-3)

        # fix design variables of the RPB
        m.fs.RPB.ads.Tx.fix()
        m.fs.RPB.des.Tx.fix()

        # Fix at design decision specified by user
        m.fs.RPB.w_rpm.fix(self.w_rpm)

        # Fix inlet pressure at decision specified by user
        m.fs.flue_gas_in.properties[0.0].pressure.setub(3e5)
        m.fs.RPB.ads.inlet_properties[0.0].pressure.setub(3e5)
        m.fs.flue_gas_in.pressure.fix(self.pressure)

        # initialize feed and product blocks
        m.fs.flue_gas_in.initialize()
        m.fs.flue_gas_out.initialize()
        m.fs.steam_sweep_feed.initialize()
        m.fs.regeneration_prod.initialize()

        # propagate feed and product blocks (for initial RPB guesses)
        propagate_state(arc = m.fs.s_flue_gas, direction="forward")
        propagate_state(arc = m.fs.s_steam_feed, direction="forward")
        propagate_state(arc = m.fs.s_cleaned_flue_gas, direction="backward")
        propagate_state(arc = m.fs.s_regeneration_prod, direction="backward")

        # Initialize RPB (about 4 mins for smaller discretization size and close to 20 mins for the larger size)
        optarg = {
            # "halt_on_ampl_error": "yes",
            "max_iter": 1000,
            "bound_push": 1e-22,
            # "mu_init": 1e-3,
            "nlp_scaling_method": "user-scaling",
        }

        init_points = [1e-5,1e-3,1e-1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

        for i in m.fs.RPB.ads.z:
            for j in m.fs.RPB.ads.o:
                m.fs.RPB.ads.gas_properties[0, i, j].pressure.setub(3e5)

        m.fs.RPB.initialize(outlvl=idaeslog.INFO, optarg=optarg, initialization_points=init_points)

        # full solve with IPOPT
        Solver = SolverFactory("ipopt")
        Solver.solve(m, tee=True).write()
        
        return m
    
    def finalize_model(self):
        """
        Currently, I put everything in the create
        model section. So this is a dead call.
        
        """
        pass

    def label_experiment(self):
        """
        Annotating (labeling) the model with experimental 
        data, associated measurement error, experimental 
        design decisions, and unknown model parameters.

        """
        m = self.model
        
        #################################
        # Labeling experiment outputs
        # (experiment measurements)
        
        m.experiment_outputs = Suffix(direction=Suffix.LOCAL)
        # Add adsorber gas temperature to outputs
        m.experiment_outputs.update((m.fs.RPB.ads.Tg[0, i, j], m.fs.RPB.ads.Tg[0, i, j]()) for i in m.fs.RPB.ads.z for j in m.fs.RPB.ads.o)
        # Add desorber gas temperature to outputs
        m.experiment_outputs.update((m.fs.RPB.des.Tg[0, i, j], m.fs.RPB.des.Tg[0, i, j]()) for i in m.fs.RPB.ads.z for j in m.fs.RPB.ads.o)
        # Add adsorber mole fraction of CO2 to outputs
        m.experiment_outputs.update((m.fs.RPB.ads.y[0, i, j, "CO2"], m.fs.RPB.ads.y[0, i, j, "CO2"]()) for i in m.fs.RPB.ads.z for j in m.fs.RPB.ads.o)
        # Add desorber mole fraction of CO2 to outputs
        m.experiment_outputs.update((m.fs.RPB.des.y[0, i, j, "CO2"], m.fs.RPB.des.y[0, i, j, "CO2"]()) for i in m.fs.RPB.ads.z for j in m.fs.RPB.ads.o)
        
        # End experiment outputs
        #################################
        
        #################################
        # Labeling unknown parameters
        
        m.unknown_parameters = Suffix(direction=Suffix.LOCAL)
        # Add labels to all unknown parameters with nominal value as the value
        # m.unknown_parameters.update((k, k.value) for k in [m.fs.RPB.C1, m.fs.RPB.hgx, m.fs.RPB.delH_1, m.fs.RPB.delH_2, m.fs.RPB.delH_3])
        # m.unknown_parameters.update((k, k.value) for k in [m.fs.RPB.C1, m.fs.RPB.delH_1, m.fs.RPB.delH_2, m.fs.RPB.delH_3])
        m.unknown_parameters.update((k, k.value) for k in [m.fs.RPB.C1, m.fs.RPB.delH_1, m.fs.RPB.delH_2, m.fs.RPB.delH_3, m.fs.RPB.ads.hgx, m.fs.RPB.des.hgx])
        

        # End unknown parameters
        #################################
        
        #################################
        # Labeling experiment inputs
        # (experiment design decisions)
        
        m.experiment_inputs = Suffix(direction=Suffix.LOCAL)
        # Add experimental input label for control variable (m.U1)
        m.experiment_inputs[m.fs.RPB.w_rpm] = None
        m.experiment_inputs[m.fs.flue_gas_in.pressure] = None
        
        # End experiment inputs
        #################################
        
        #################################
        # Labeling measurement error
        # (for experiment outputs)
        
        m.measurement_error = Suffix(direction=Suffix.LOCAL)
        # Add adsorber gas temperature to measurement error (assume 0.5 degree measurement error)
        m.measurement_error.update((m.fs.RPB.ads.Tg[0, i, j], 0.5) for i in m.fs.RPB.ads.z for j in m.fs.RPB.ads.o)
        # Add desorber gas temperature to measurement error (assume 0.5 degree measurement error)
        m.measurement_error.update((m.fs.RPB.des.Tg[0, i, j], 0.5) for i in m.fs.RPB.ads.z for j in m.fs.RPB.ads.o)
        # Add adsorber mole fraction of CO2 to measurement error, proportional 1 percent error
        m.measurement_error.update((m.fs.RPB.ads.y[0, i, j, "CO2"], 0.01*m.fs.RPB.ads.y[0, i, j, "CO2"]()) for i in m.fs.RPB.ads.z for j in m.fs.RPB.ads.o)
        # Add desorber mole fraction of CO2 to measurement error, proportional 1 percent error
        m.measurement_error.update((m.fs.RPB.des.y[0, i, j, "CO2"], 0.01*m.fs.RPB.des.y[0, i, j, "CO2"]()) for i in m.fs.RPB.ads.z for j in m.fs.RPB.ads.o)
        
        
        # End measurement error
        #################################
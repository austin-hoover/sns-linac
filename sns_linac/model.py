import math
import os
import pathlib
from typing import Any
from typing import Callable

import numpy as np

from orbit.core.bunch import Bunch
from orbit.core.linac import BaseRfGap
from orbit.core.linac import MatrixRfGap
from orbit.core.linac import RfGapTTF
from orbit.core.spacecharge import SpaceChargeCalc3D
from orbit.core.spacecharge import SpaceChargeCalcUnifEllipse
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.py_linac.lattice import AxisFieldRF_Gap
from orbit.py_linac.lattice import AxisField_and_Quad_RF_Gap
from orbit.py_linac.lattice import BaseLinacNode
from orbit.py_linac.lattice import BaseRF_Gap
from orbit.py_linac.lattice import Bend
from orbit.py_linac.lattice import Drift
from orbit.py_linac.lattice import LinacApertureNode
from orbit.py_linac.lattice import LinacEnergyApertureNode
from orbit.py_linac.lattice import LinacPhaseApertureNode
from orbit.py_linac.lattice import LinacTrMatrixGenNode
from orbit.py_linac.lattice import OverlappingQuadsNode 
from orbit.py_linac.lattice import Quad
from orbit.py_linac.lattice_modifications import Add_drift_apertures_to_lattice
from orbit.py_linac.lattice_modifications import Add_quad_apertures_to_lattice
from orbit.py_linac.lattice_modifications import Add_rfgap_apertures_to_lattice
from orbit.py_linac.lattice_modifications import AddMEBTChopperPlatesAperturesToSNS_Lattice
from orbit.py_linac.lattice_modifications import AddScrapersAperturesToLattice
from orbit.py_linac.lattice_modifications import GetLostDistributionArr
from orbit.py_linac.lattice_modifications import Replace_BaseRF_Gap_and_Quads_to_Overlapping_Nodes
from orbit.py_linac.lattice_modifications import Replace_BaseRF_Gap_to_AxisField_Nodes
from orbit.py_linac.lattice_modifications import Replace_Quads_to_OverlappingQuads_Nodes
from orbit.py_linac.linac_parsers import SNS_LinacLatticeFactory
from orbit.py_linac.overlapping_fields import SNS_EngeFunctionFactory
from orbit.space_charge.sc3d import setSC3DAccNodes
from orbit.space_charge.sc3d import setUniformEllipsesSCAccNodes


SEQUENCES = [
    "MEBT",
    "DTL1",
    "DTL2",
    "DTL3",
    "DTL4",
    "DTL5",
    "DTL6",
    "CCL1",
    "CCL2",
    "CCL3",
    "CCL4",
    "SCLMed",
    "SCLHigh",
    "HEBT1",
    "HEBT2",
]


def add_aperture_nodes_to_classes(
    lattice: AccLattice,
    classes: list = None,
    nametag: str = "aprt",
    node_constructor: Callable = None,
    node_constructor_kws: dict = None,
) -> list[AccNode]:
    """Add aperture nodes to all nodes of a specified class (or classes).

    Parameters
    ----------
    lattice: AccLattice
        The accelerator lattice.
    classes : list
        Add child node to parent if parent's class is in this list.
    nametag : str
        Nodes are named "{parent_node_name}_{nametag}_in" and "{parent_node_name}_{nametag}_out".
    node_constructor : callable
        Returns an aperture node.
    node_constructor_kws : dict
        Key word arguments for `node_constructor`. (`aperture_node = node_constructor(**node_constructor)`).

    Returns
    -------
    list[AccNode]
        The aperture nodes added to the lattice.
    """
    node_pos_dict = lattice.getNodePositionsDict()
    aperture_nodes = []
    for node in lattice.getNodesOfClasses(classes):
        if node.hasParam("aperture") and node.hasParam("aprt_type"):
            for location, suffix, position in zip(
                [node.ENTRANCE, node.EXIT], ["in", "out"], node_pos_dict[node]
            ):
                aperture_node = node_constructor(**node_constructor_kws)
                aperture_node.setName(f"{node.getName()}_{nametag}_{suffix}")
                aperture_node.setPosition(position)
                aperture_node.setSequence(node.getSequence())
                node.addChildNode(aperture_node, location)
                aperture_nodes.append(aperture_node)
    return aperture_nodes


def add_aperture_nodes_to_drifts(
    lattice: AccLattice,
    start: float = 0.0,
    stop: float = None,
    step: float = 1.0,
    nametag: str = "aprt",
    node_constructor: Callable = None,
    node_constructor_kws: dict = None,
) -> list[AccNode]:
    """Add aperture nodes to drift spaces as child nodes.

    Parameters
    ----------
    lattice: AccLattice
        The accelerator lattice.
    start, stop, stop. : float
        Nodes are added between `start` [m] and `stop` [m] with spacing `step` [m].
    nametag : str
        Nodes are named "{parent_node_name}:{part_index}_{nametag}".
    node_constructor : callable
        Returns an aperture node.
    node_constructor_kws : dict
        Key word arguments for `node_constructor`. (`aperture_node = node_constructor(**node_constructor)`).

    Returns
    -------
    list[AccNode]
        The aperture nodes added to the lattice.
    """
    if node_constructor is None:
        return

    if node_constructor_kws is None:
        node_constructor_kws = dict()

    if stop is None:
        stop = lattice.getLength()

    node_pos_dict = lattice.getNodePositionsDict()
    parent_nodes = lattice.getNodesOfClasses([Drift])
    last_position, _ = node_pos_dict[parent_nodes[0]]
    last_position = last_position - 2.0 * step
    child_nodes = []
    for parent_node in parent_nodes:
        position, _ = node_pos_dict[parent_node]
        if position > stop:
            break
        for index in range(parent_node.getnParts()):
            if start <= position <= stop:
                if position >= (last_position + step):
                    child_node = node_constructor(**node_constructor_kws)
                    name = "{}".format(parent_node.getName())
                    if parent_node.getnParts() > 1:
                        name = "{}:{}".format(name, index)
                    child_node.setName("{}_{}".format(name, nametag))
                    child_node.setPosition(position)
                    child_node.setSequence(parent_node.getSequence())
                    parent_node.addChildNode(
                        child_node, parent_node.BODY, index, parent_node.BEFORE
                    )
                    child_nodes.append(child_node)
                    last_position = position
            position += parent_node.getLength(index)
    return child_nodes


def make_phase_aperture_node(phase_min: float, phase_max: float, rf_freq: float) -> LinacPhaseApertureNode:
    aperture_node = LinacPhaseApertureNode(frequency=rf_freq)
    aperture_node.setMinMaxPhase(phase_min, phase_max)
    return aperture_node


def make_energy_aperture_node(energy_min: float, energy_max: float) -> LinacEnergyApertureNode:
    aperture_node = LinacEnergyApertureNode()
    aperture_node.setMinMaxEnergy(energy_min, energy_max)
    return aperture_node
    
    
class AccModel:
    def __init__(self, verbose: int = 1) -> None:
        self.verbose = verbose


class SNS_LINAC(AccModel):
    def __init__(
        self, 
        xml_filename: str = None,
        sequence_start: str = "MEBT",
        sequence_stop: str = "HEBT2",
        max_drift: float = 0.010, 
        rf_freq: float = 402.5e+06,
        verbose: bool = True,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        
        self.path = pathlib.Path(__file__)
        self.xml_filename = xml_filename
        if self.xml_filename is None:
            self.xml_filename = os.path.join(self.path.parent, "data/sns_linac.xml")

        index_start = SEQUENCES.index(sequence_start)
        index_stop = SEQUENCES.index(sequence_stop)
        self.sequences = SEQUENCES[index_start : index_stop + 1]
        
        sns_linac_factory = SNS_LinacLatticeFactory()
        if max_drift:
            sns_linac_factory.setMaxDriftLength(max_drift)
    
        self.lattice = sns_linac_factory.getLinacAccLattice(self.sequences, self.xml_filename)

        self.rf_freq = rf_freq
        self.verbose = verbose
        self.aperture_nodes = []
        self.sc_nodes = []

        if verbose:
            print("Initialized lattice")
            print("xml file = {}".format(self.xml_filename))
            print("lattice length = {:.3f} [m])".format(self.lattice.getLength()))

    def set_rf_gap_model(self, name: str = "ttf") -> None:            
        rf_gap_model_constructors = {
            "base": BaseRfGap,
            "matrix": MatrixRfGap,
            "ttf": RfGapTTF,
        }
        rf_gap_model_constructor = rf_gap_model_constructors[name]
        for rf_gap_node in self.lattice.getRF_Gaps():
            rf_gap_model = rf_gap_model_constructor()
            rf_gap_node.setCppGapModel(rf_gap_model)

    def add_aperture_nodes(
        self, 
        scrape_x: float = 0.042, 
        scrape_y: float = 0.042
    ) -> list[AccNode]:
        """Add aperture nodes to quads, rf gaps, mebt choppers, and scrapers."""
        aperture_nodes = Add_quad_apertures_to_lattice(self.lattice)
        aperture_nodes = Add_rfgap_apertures_to_lattice(self.lattice, aperture_nodes)
        aperture_nodes = AddMEBTChopperPlatesAperturesToSNS_Lattice(self.lattice, aperture_nodes)        
        aperture_nodes = AddScrapersAperturesToLattice(
            self.lattice, "MEBT_Diag:H_SCRP", scrape_x, scrape_y, aperture_nodes
        )    
        aperture_nodes = AddScrapersAperturesToLattice(
            self.lattice, "MEBT_Diag:V_SCRP", scrape_x, scrape_y, aperture_nodes
        )
        self.aperture_nodes.extend(aperture_nodes)
        return aperture_nodes

    def add_aperture_nodes_to_drifts(
        self, 
        start: float = 0.0, 
        stop: float = None, 
        step: float = 1.0, 
        radius: float = 0.042, 
    ) -> list[AccNode]:
        """Add circular apertures in drifts between start and stop position."""
        if stop is None:
            stop = self.lattice.getLength()
        aperture_nodes = Add_drift_apertures_to_lattice(self.lattice, start, stop, step, 2.0 * radius)
        self.aperture_nodes.extend(aperture_nodes)
        return aperture_nodes

    def add_phase_aperture_nodes(
        self,
        phase_min: float = -180.0,  # [deg]
        phase_max: float = +180.0,  # [deg]
        classes: list = None,
        drifts: bool = False,
        drift_start: float = 0.0,
        drift_stop: float = None,
        drift_step: float = 1.0,
        nametag: str = "phase_aprt",
    ) -> list[AccNode]:
        """Add longitudinal phase aperture nodes."""
        if classes is None:
            classes = [
                BaseRF_Gap, 
                AxisFieldRF_Gap, 
                AxisField_and_Quad_RF_Gap,
                Quad, 
                OverlappingQuadsNode,
            ]
            
        node_constructor = make_phase_aperture_node
        node_constructor_kws = {
            "phase_min": phase_min,
            "phase_max": phase_max,
            "rf_freq": self.rf_freq,
        }          
        if classes:
            aperture_nodes = add_aperture_nodes_to_classes(
                lattice=self.lattice,
                classes=classes,
                nametag=nametag,
                node_constructor=node_constructor,
                node_constructor_kws=node_constructor_kws,
            )
            self.aperture_nodes.extend(aperture_nodes)            
        if drifts:
            aperture_nodes = add_aperture_nodes_to_drifts(
                lattice=self.lattice,
                start=drift_start,
                stop=drift_stop,
                step=drift_step,
                nametag=nametag,
                node_constructor=node_constructor,
                node_constructor_kws=node_constructor_kws,
            )
            self.aperture_nodes.extend(aperture_nodes)                
        return aperture_nodes

    def add_energy_aperture_nodes(
        self,
        energy_min: float = -1.000,  # [GeV]
        energy_max: float = -1.000,  # [GeV]
        classes: list = None,
        drifts: bool = False,
        drift_start: float = 0.0,
        drift_stop: float = None,
        drift_step: float = 1.0,
        nametag: str = "phase_aprt",
    ) -> list[AccNode]:
        """Add longitudinal phase aperture nodes."""
        if classes is None:
            classes = [
                BaseRF_Gap, 
                AxisFieldRF_Gap, 
                AxisField_and_Quad_RF_Gap,
                Quad, 
                OverlappingQuadsNode,
            ]
            
        node_constructor = make_energy_aperture_node
        node_constructor_kws = {
            "energy_min": energy_min,
            "energy_max": energy_max,
        }          
        if classes:
            aperture_nodes = add_aperture_nodes_to_classes(
                lattice=self.lattice,
                classes=classes,
                nametag=nametag,
                node_constructor=node_constructor,
                node_constructor_kws=node_constructor_kws,
            )
            self.aperture_nodes.extend(aperture_nodes)            
        if drifts:
            aperture_nodes = add_aperture_nodes_to_drifts(
                lattice=self.lattice,
                start=drift_start,
                stop=drift_stop,
                step=drift_step,
                nametag=nametag,
                node_constructor=node_constructor,
                node_constructor_kws=node_constructor_kws,
            )
            self.aperture_nodes.extend(aperture_nodes)                
        return aperture_nodes

    def add_sc_nodes(
        self,
        solver: str = "fft",
        gridx: int = 64,
        gridy: int = 64,
        gridz: int = 64,
        path_length_min: float = 0.010,
        n_ellipsoids: int = 5,
    ) -> list[AccNode]:
        sc_nodes = []
        if solver == "fft":
            sc_calc = SpaceChargeCalc3D(gridx, gridy, gridz)
            sc_nodes = setSC3DAccNodes(self.lattice, path_length_min, sc_calc)
        elif solver == "ellipsoid":
            sc_calc = SpaceChargeCalcUnifEllipse(n_ellipsoids)
            sc_nodes = setUniformEllipsesSCAccNodes(self.lattice, path_length_min, sc_calc)
        else:
            raise ValueError(f"Invalid spacecharge solver {solver}")

        if self.verbose:
            lengths = [node.getLengthOfSC() for node in sc_nodes]
            min_length = min(min(lengths), self.lattice.getLength())
            max_length = max(max(lengths), 0.0)
            
            print(f"Added {len(sc_nodes)} space charge nodes (solver={solver})")
            print(f"min sc node length = {min_length}".format(min_length))
            print(f"max sc node length = {min_length}".format(max_length))
            
        self.sc_nodes = sc_nodes
        return sc_nodes

    def set_overlapping_rf_and_quad_fields(
        self, 
        sequences: list[str] = None, 
        z_step: float = 0.002,
        cav_names: list[str] = None,
        fields_dir: str = None,
        use_longitudinal_quad_field: bool = True,
    ) -> None:
        """Replace overlapping quad/rf nodes in specified sequences."""
        if fields_dir is None:
            fields_dir = os.path.join(self.path.parent, "data/sns_rf_fields/")
            
        if sequences is None:
            sequences = self.sequences
        sequences = sequences.copy()
        sequences = [seq for seq in sequences if seq not in ["HEBT1", "HEBT2"]]

        if cav_names is None:
            cav_names = []
            
        # Replace hard-edge quads with soft-edge quads; replace zero-length RF gap models
        # with field-on-axis RF gap models. Can be used for any sequences, no limitations.
        Replace_BaseRF_Gap_and_Quads_to_Overlapping_Nodes(
            self.lattice, z_step, fields_dir, sequences, cav_names, SNS_EngeFunctionFactory
        )

        # Add tracking through the longitudinal field component of the quad. The
        # longitudinal component is nonzero only for the distributed magnetic field
        # of the quad. 
        for node in self.lattice.getNodes():
            if (isinstance(node, OverlappingQuadsNode) or isinstance(node, AxisField_and_Quad_RF_Gap)):
                node.setUseLongitudinalFieldOfQuad(use_longitudinal_quad_field)

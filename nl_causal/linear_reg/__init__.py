from .sparse_reg import WLasso, SCAD, SCAD_IC, L0_IC
import sys
sys.path.append('../..')
sys.path.append('..')

__all__ = [
	"WLasso", 
	"SCAD",
	"SCAD_IC",
	"L0_IC"
	]
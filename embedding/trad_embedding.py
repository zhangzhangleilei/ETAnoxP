import os
import sys
import pandas as pd
import numpy as np
from modlamp.sequences import Random
import difflib
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor



def main():
    save_path = args.save_path
    input_path = args.input_path
    for file in os.listdir(input_path):
        if file.endswith(".fasta"):
            pepdesc_gi = PeptideDescriptor(imput_path + file, 'eisenberg')
            globdesc_gi = GlobalDescriptor(imput_path + file)
            pepdesc_gi.calculate_global()  # calculate global Eisenberg hydrophobicity
            pepdesc_gi.calculate_moment(append=True)
            pepdesc_gi.load_scale('gravy')  # load GRAVY scale
            pepdesc_gi.calculate_global(append=True)  # calculate global GRAVY hydrophobicity
            pepdesc_gi.calculate_moment(append=True)  # calculate GRAVY hydrophobic moment
            pepdesc_gi.load_scale('z3')  # load old Z scale
            pepdesc_gi.calculate_autocorr(1, append=True)  # calculate global Z scale (=window1 autocorrelation)
            pepdesc_gi.load_scale('z5')  # load old Z scale
            pepdesc_gi.calculate_autocorr(1, append=True)  # calculate global Z scale (=window1 autocorrelation)
            # pepdesc_gi.load_scale('AASI')
            # pepdesc_gi.calculate_global(append=True)  # calculate global AASI index
            # pepdesc_gi.calculate_moment(append=True)  # calculate AASI index moment
            pepdesc_gi.load_scale('ABHPRK')
            pepdesc_gi.calculate_global(append=True)  # calculate ABHPRK feature
            pepdesc_gi.load_scale('argos')
            pepdesc_gi.calculate_global(append=True)  # calculate global argos index
            pepdesc_gi.calculate_moment(append=True)  # calculate argos index moment
            pepdesc_gi.load_scale('bulkiness')
            pepdesc_gi.calculate_global(append=True)  # calculate global bulkiness index
            pepdesc_gi.calculate_moment(append=True)  # calculate bulkiness index moment
            pepdesc_gi.load_scale('charge_phys')
            pepdesc_gi.calculate_global(append=True)  # calculate global charge_phys index
            pepdesc_gi.load_scale('charge_acid')
            pepdesc_gi.calculate_global(append=True)  # calculate global charge_acid index
            # pepdesc_gi.load_scale('Ez')
            # pepdesc_gi.calculate_global(append=True)  # calculate global energies of insertion of amino acid side chains into lipid bilayers index
            # pepdesc_gi.load_scale('flexibility')
            # pepdesc_gi.calculate_global(append=True)  # calculate global flexibility scale
            # pepdesc_gi.calculate_moment(append=True)  # calculate flexibility moment
            pepdesc_gi.load_scale('grantham')
            pepdesc_gi.calculate_global(
                append=True)  # calculate global amino acid side chain composition, polarity and molecular volume
            pepdesc_gi.load_scale('hopp-woods')
            pepdesc_gi.calculate_global(append=True)  # calculate global Hopp-Woods hydrophobicity scale
            pepdesc_gi.calculate_moment(append=True)  # calculate Hopp-Woods hydrophobicity moment
            pepdesc_gi.load_scale('ISAECI')
            pepdesc_gi.calculate_global(
                append=True)  # calculate global ISAECI (Isotropic Surface Area (ISA) and Electronic Charge Index (ECI) of amino acid side chains) index
            pepdesc_gi.load_scale('janin')
            pepdesc_gi.calculate_global(append=True)  # calculate global Janin hydrophobicity scale
            pepdesc_gi.calculate_moment(append=True)  # calculate Janin hydrophobicity moment
            pepdesc_gi.load_scale('kytedoolittle')
            pepdesc_gi.calculate_global(append=True)  # calculate global Kyte & Doolittle hydrophobicity scale
            pepdesc_gi.calculate_moment(append=True)  # calculate Kyte & Doolittle hydrophobicity moment
            # pepdesc_gi.load_scale('levitt_alpha')
            # pepdesc_gi.calculate_global(append=True)  # calculate global Levitt alpha-helix propensity scale
            # pepdesc_gi.calculate_moment(append=True)  # calculate Levitt alpha-helix propensity moment
            # pepdesc_gi.load_scale('MSS')
            # pepdesc_gi.calculate_global(append=True)  # calculate global MSS index, graph-theoretical index that reflects topological shape and size of amino acid side chains
            # pepdesc_gi.calculate_moment(append=True)  # calculate MSS moment
            pepdesc_gi.load_scale('MSW')
            pepdesc_gi.calculate_global(
                append=True)  # calculate global MSW scale, Amino acid scale based on a PCA of the molecular surface based WHIM descriptor (MS-WHIM), extended to natural amino acids
            # pepdesc_gi.load_scale('pepArc')
            # pepdesc_gi.calculate_global(append=True) # calculate global pepArc, modlabs pharmacophoric feature scale, dimensions are: hydrophobicity, polarity, positive charge, negative charge, proline.
            pepdesc_gi.load_scale('pepcats')
            pepdesc_gi.calculate_global(
                append=True)  # calculate global pepcats, modlabs pharmacophoric feature based PEPCATS scale
            pepdesc_gi.load_scale('polarity')
            pepdesc_gi.calculate_global(append=True)  # calculate global AA polarity
            pepdesc_gi.calculate_moment(append=True)  # calculate AA polarity moment
            pepdesc_gi.load_scale('PPCALI')
            pepdesc_gi.calculate_global(
                append=True)  # calculate global modlabs inhouse scale derived from a PCA of 143 amino acid property scales
            # pepdesc_gi.load_scale('refractivity')
            # pepdesc_gi.calculate_global(append=True) # calculate global relative AA refractivity
            # pepdesc_gi.calculate_moment(append=True) # calculate relative AA refractivity moment
            pepdesc_gi.load_scale('t_scale')
            pepdesc_gi.calculate_global(
                append=True)  # calculate global t scale, A PCA derived scale based on amino acid side chain properties calculated with 6 different probes of the GRID program
            # pepdesc_gi.load_scale('TM_tend')
            # pepdesc_gi.calculate_global(append=True) # calculate global Amino acid transmembrane propensity scale
            # pepdesc_gi.calculate_moment(append=True) # calculate Amino acid transmembrane propensity scale moment

            # col_names1 = 'ID,Sequence,H_Eisenberg,uH_Eisenberg,H_GRAVY,uH_GRAVY,Z3_1,Z3_2,Z3_3, Z5_1,Z5_2,Z5_3, Z5_4,Z5_5,S_AASI, uS_AASI, modlas_ABHPRK, H_argos, uH_argos, B_Builkiness, uB_Builkiness, charge_phys, charge_acid, Ez, flexibility, u_flexibility, Grantham, H_HoppWoods, uH-HoppWoods, ISAECI, H_Janin, uH_Janin, H_KyteDoolittle, uH_KyteDoolittle, F_Levitt, uF_Levitt, MSS_shape, u_MSS_shape, MSW, pepArc, pepcats, polarity, u_polarity, PPCALI, refractivity, u_refractivity, t_scale, TM_tend, u_TM_tend'
            col_names1 = 'ID,Sequence,H_Eisenberg,uH_Eisenberg,H_GRAVY,uH_GRAVY,Z3_1,Z3_2,Z3_3, Z5_1,Z5_2,Z5_3, Z5_4,Z5_5, modlas_ABHPRK, H_argos, uH_argos, B_Builkiness, uB_Builkiness, charge_phys, charge_acid, Grantham, H_HoppWoods, uH-HoppWoods, ISAECI, H_Janin, uH_Janin, H_KyteDoolittle, uH_KyteDoolittle, MSW, pepcats, polarity, u_polarity, PPCALI,  t_scale'

            pepdesc_gi.save_descriptor(save_path + '{}_moldamp_pepfea.csv'.format(file.split(".")[0]),
                                       header=col_names1)
            print(file, "pep-fea-over!")
            globdesc_gi.length()  # sequence length
            globdesc_gi.boman_index(append=True)  # Boman index
            globdesc_gi.aromaticity(append=True)  # global aromaticity
            globdesc_gi.aliphatic_index(append=True)  # aliphatic index
            globdesc_gi.instability_index(append=True)  # instability index
            globdesc_gi.calculate_charge(ph=7.4, amide=False, append=True)  # net charge
            globdesc_gi.calculate_MW(amide=False, append=True)  # molecular weight
            globdesc_gi.isoelectric_point(amide=False, append=True)  # isoelectric point
            globdesc_gi.hydrophobic_ratio(append=True)

            col_names2 = 'ID, Sequence,Length,BomanIndex,Aromaticity,AliphaticIndex,InstabilityIndex, NetCharge, MW, IsoelectricPoint, HydrophobicRatio'

            # col_names2 = 'ID, Sequence,InstabilityIndex, NetCharge, MW, IsoelectricPoint, HydrophobicRatio'
            globdesc_gi.save_descriptor(save_path + '{}_moldamp_gloabfea.csv'.format(file.split(".")[0]),
                                        header=col_names2)
            print(file, "gloab-fea-over!")

            pepdesc_gi = pd.read_csv(save_path + '{}_moldamp_pepfea.csv'.format(file.split(".")[0]), index_col=0)
            globdesc_gi = pd.read_csv(save_path + '{}_moldamp_gloabfea.csv'.format(file.split(".")[0]), index_col=0)

            generated_inliers_props = pepdesc_gi.join(globdesc_gi)
            # Removing duplicated columns
            generated_inliers_props = generated_inliers_props.drop(['Sequence', ' Sequence'], axis=1)
            # merage peptide and global
            generated_inliers_props.to_csv(save_path + '{}_moldamp_features.csv'.format(file.split(".")[0]),
                                           index_label=False)
            print(file, "merge-fea-over!")
def get_finetune_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str,
                        default="./",
                        help='save_path path')
    parser.add_argument('--input_path', type=str,
                        default="./",
                        help='input_path')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args=get_finetune_config()
    main()
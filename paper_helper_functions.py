"""
I want to maintain the `prep_paper_figures.ipynb` notebook as clean and readable as possible and so
some things will be moved into this script for this purpose.
"""
import pandas as pd
import Levenshtein as Lev
import numpy as np
import re
import os
import math

def rename_paralogs(vgene: str):
    """ Rename antibody variable genes to be simpler and consider paralogs as the same.
    Args:
        vgene (str) Variable gene name likely of the form "IGHV3-30D (Human)"
    Returns:
        gene (str) Simplified name of the input vgene such as "IGHV3-30"
    """

    try:
        gene, species = vgene.split(" ")
    except:
        return False

    if gene[-1] == "D":
        gene = gene[:-1]
    return gene


def is_human(vgene: str):
    """ Return True if the variable gene has the word human in it.
    Args:
        vgene (str) Variable gene name likely of the form "IGHV3-30D (Human)"
    Returns:
        bool - True if the gene ended with "(Human)"
    """
    try:
        gene, species = vgene.split(" ")
        return species == "(Human)"
    except:
        return False


def prep_human_covabdab(dab_fname: str, treat_paralogs_same=True):
    """Prep the CoVAbDab dataset for use in the paper.

    Args:
        dab_fname (str) Path to the covabdab csv file.
        treat_paralogs_same (bool) Whether or not to treat variable gene paralogs as the same gene
    Returns:
        covdf (pandas.dataframe)
    """
    covdf = pd.read_csv(dab_fname)

    covdf["Human"] = covdf["Heavy V Gene"].apply(is_human)
    covdf = covdf[covdf["Human"]]

    if treat_paralogs_same:
        covdf["HV"] = covdf["Heavy V Gene"].apply(rename_paralogs)
        covdf["LV"] = covdf["Light V Gene"].apply(rename_paralogs)


    covdf = covdf[covdf["LV"].apply(lambda x: x != False)]
    covdf = covdf.dropna(subset=['HV', 'LV', 'CDRH3', 'CDRL3'])

    covdf = covdf.reset_index()

    return covdf


def get_similar_seq_feature_pts(hv, lv, cdrh3, cdrl3, refdf, cdrh3_cutoff=0.5, cdrl3_cutoff=0.5, ref_name=""):
    """Get collection of points in a reference df that have similar sequence features to a reference mab.

    Args:
        hv (str): Heavy chain V gene
        lv (str): Light chain V gene
        cdrh3 (str): Heavy chain CDR3 sequence
        cdrl3 (str): Light chain CDR3 sequence
        ref_name (str): Name of the antibody being screened against the database.
            If provided then ignore this name in the refdf.
        refdf (pd.DataFrame): Dataframe of reference antibodies. Must contain columns ["HV", "LV", "CDRH3", "CDRL3"]
        cdrh3_cutoff (float): Minimum identity of CDRH3 to be considered a pub clone
        cdrl3_cutoff (float): Minimum identity of CDRL3 to be considered a pub clone
    Returns:
        pts (list): List of tuples of the form (CDRH3_ID, CDRL3_ID, CGroup) where CGroup is the color category
        num_h_blue (int): Number of heavy chains that are blue
        num_l_blue (int): Number of light chains that are blue
        df (pd.DataFrame): Dataframe of all points that were plotted
    """
    # Now compare that reference antibody to every antibody in the Light Chain Coherence Datasets
    # color categories are 0 = different V genes, 1 = same light, 2 = same heavy, 3 = same both
    pts = []  # will store tuples of the form (hid, lid, color_category)
    # Info I need is: HV+ LV+ CDRH3+ CDRL3+
    df = refdf.copy()
    # Filter out antibodies with the same name or these 3 special cases
    # of essentially the same antibody
    if ref_name:
        df = df[df["Name"] != ref_name]
    if ref_name == "76E1":
        df = df[df["Name"] != "Sun_76E1"]
    elif ref_name == "DH1058":
        df = df[df["Name"] != "DH1058-1"]
        df = df[df["Name"] != "H711725+K711414"]

    df["HV+"] = df["HV"] == hv
    df["LV+"] = df["LV"] == lv
    df["CDRH3_ID"] = df["CDRH3"].apply(lambda x: get_fract_identity(x, cdrh3))
    df["CDRL3_ID"] = df["CDRL3"].apply(lambda x: get_fract_identity(x, cdrl3))
    df["CDRH3+"] = df["CDRH3_ID"] >= cdrh3_cutoff
    df["CDRL3+"] = df["CDRL3_ID"] >= cdrl3_cutoff

    # Get color group
    color_conditions = [(df["HV+"] & df["LV+"]),  # 3
                        df["HV+"],  # 2
                        df["LV+"],  # 1
                        df["CDRH3+"] | df["CDRL3+"]]  # 0

    corresponding_groups = [3, 2, 1, 0]
    df["CGroup"] = np.select(color_conditions, corresponding_groups, default=-1)

    # final is pts list which is a tuple of hid, lid, cgroup
    df = df[df["CGroup"] != -1]

    df = df.sort_values(by='CGroup', ascending=True)
    num_h_blue = len(df[df["CDRH3+"] & (df["CGroup"] == 3)])
    num_l_blue = len(df[df["CDRL3+"] & (df["CGroup"] == 3)])

    pts = df[["CDRH3_ID", "CDRL3_ID", "CGroup"]].to_records(index=False).tolist()

    return pts, num_h_blue, num_l_blue, df


def get_fract_identity(str1, str2):
    return 1 - (Lev.distance(str1, str2) / float(max(len(str1), len(str2))))


## DSSP Things
class DSSPData:
    def __init__(self):
        self.num = []
        self.resnum = []
        self.inscode = []
        self.chain = []
        self.aa = []
        self.struct = []
        self.bp1 = []
        self.bp2 = []
        self.acc = []
        self.h_nho1 = []
        self.h_ohn1 = []
        self.h_nho2 = []
        self.h_ohn2 = []
        self.tco = []
        self.kappa = []
        self.alpha = []
        self.phi = []
        self.psi = []
        self.xca = []
        self.yca = []
        self.zca = []

    def parseDSSP(self, file):
        input_handle = open(file, 'r')

        line_num = 0
        start = False
        for line in input_handle:

            if (re.search('#', line)):
                start = True
                continue

            if (start):
                self.num.append(line[0:5].strip())
                self.resnum.append(line[5:10].strip())
                self.inscode.append(line[10:11].strip())
                self.chain.append(line[11:12].strip())
                self.aa.append(line[12:14].strip())
                self.struct.append(line[16:25])
                self.bp1.append(line[25:29].strip())
                self.bp2.append(line[29:34].strip())
                self.acc.append(line[34:38].strip())
                self.h_nho1.append(line[38:50].strip())
                self.h_ohn1.append(line[50:61].strip())
                self.h_nho2.append(line[61:72].strip())
                self.h_ohn2.append(line[72:83].strip())
                self.tco.append(line[83:91].strip())
                self.kappa.append(line[91:97].strip())
                self.alpha.append(line[97:103].strip())
                self.phi.append(line[103:109].strip())
                self.psi.append(line[109:115].strip())
                self.xca.append(line[115:122].strip())
                self.yca.append(line[122:129].strip())
                self.zca.append(line[129:136].strip())

    def getResnums(self):
        return self.resnum

    def getResnumi(self):
        return [str(resnum) + i for resnum, i in zip(self.getResnums(), self.getInsCode())]

    def getInsCode(self):
        return self.inscode

    def getChain(self):
        return self.chain

    def getAAs(self, chain=''):
        if not chain:
            return self.aa
        else:
            return [aa for aa, cur_chain in zip(self.aa, self.chain) if cur_chain == chain]

    def getSecStruc(self):
        """
        The STRUCTURE columns contains a lot of information.

        Here are the definitions from column one:
        B - residue in isolated beta bridge
        E - extended strand in beta ladder
        G - 3/10 helix
        H - alpha helix
        I - 5-helix / pi helix
        T - hydrogen bonded turn
        S - bend
        """
        return self.struct

    def getBP1(self):
        return self.bp1

    def getBP2(self):
        return self.bp2

    def getACC(self):
        return self.acc

    def getH_NHO1(self):
        return self.h_nho1

    def getH_NHO2(self):
        return self.h_nho2

    def getH_OHN1(self):
        return self.h_ohn1

    def getH_OHN2(self):
        return self.h_ohn2

    def getTCO(self):
        return self.tco

    def getKAPPA(self):
        return self.kappa

    def getALPHA(self):
        return self.alpha

    def getPHI(self):
        return self.phi

    def getPSI(self):
        return self.psi

    def getX(self):
        return self.xca

    def getY(self):
        return self.yca

    def getZ(self):
        return self.zca

    def get_resnum_aa_sasa_tuples(self, chain='', ignore_cys_pairs=False):
        """ Returns list of tuples of the (resnumber, amino acid type, and rsa value) for each amino acid in `chain`.

            Only meant to be used for one chain at a time. Remember that capitalization matters.
        """
        tuples = []
        for sasa, resnum, aa, cur_chain in zip(self.getACC(), self.getResnums(), self.aa, self.chain):
            if ignore_cys_pairs and aa.islower():
                aa = "C"

            if chain:
                if cur_chain and cur_chain == chain:
                    tuples.append((resnum, aa, sasa))

            else:
                tuples.append((resnum, aa, sasa))
        return tuples

    def get_resnumi_aa_sasa_ss_tuples(self, chain='', ignore_cys_pairs=False):
        """ Returns list of tuples of the (resnumber, amino acid type, and rsa value) for each amino acid in `chain`.

            Only meant to be used for one chain at a time. Remember that capitalization matters.
        """
        tuples = []
        for sasa, i, resnum, aa, cur_chain, ss in zip(self.getACC(), self.getInsCode(), self.getResnums(), self.aa, self.chain, self.getSecStruc()):
            if ignore_cys_pairs and aa.islower():
                aa = "C"
            if i == " ":
                i = ""
            if chain:
                if cur_chain and cur_chain == chain:

                    tuples.append((str(resnum) + i, aa, sasa, ss[0]))

            else:
                tuples.append((str(resnum) + i, aa, sasa, ss[0]))
        return tuples


def get_buried_res(apo_dsspf, complex_dsspf, apo_chains, ignore_cys_pairs=True):
    """Get residues exposed in the apo structure, but buried in the complex. Returns all residues with 0s in unburied res

    Args:
        apo_dsspf: (string) path to apo dssp file
        complex_dsspf: (string) path to complex dssp file
        apo_chains: (list of strings) ex: ["L", "H"] or ["A"]. Chains to find buried residues in
        output_csv_fname: (str, opt). If present than write csv to this filename otherwise don't write it out

    Returns:
        interf_res: (list of (int, str) tuples

    """
    bsa_values = []  # Here store interface res nums

    # Create Parser objects for each
    apo = DSSPData()
    apo.parseDSSP(apo_dsspf)
    comp = DSSPData()
    comp.parseDSSP(complex_dsspf)

    # Get relative solvent accessibility for the apo chains
    for chain in apo_chains:
        apo_sasas = apo.get_resnum_aa_sasa_tuples(chain=chain, ignore_cys_pairs=ignore_cys_pairs)
        complex_sasas = comp.get_resnum_aa_sasa_tuples(chain=chain, ignore_cys_pairs=ignore_cys_pairs)

        # Make sure the residue numbers and amino acid types are the same
        assert([(n, a) for n, a, _ in apo_sasas] == [(n, a) for n, a, _ in complex_sasas])

        # Loop over the sasa values and store residues with a change in SASA
        for apo_tup, comp_tup in zip(apo_sasas, complex_sasas):
            resn = apo_tup[0]
            aa = apo_tup[1]
            delta_sasa = int(apo_tup[2]) - int(comp_tup[2])
            bsa_values.append((resn, apo_tup[1], delta_sasa))

    return bsa_values

def get_buried_resi(apo_dsspf, complex_dsspf, apo_chains, ignore_cys_pairs=True):
    """Get residues exposed in the apo structure, but buried in the complex. Returns all residues with 0s in unburied res

    Args:
        apo_dsspf: (string) path to apo dssp file
        complex_dsspf: (string) path to complex dssp file
        apo_chains: (list of strings) ex: ["L", "H"] or ["A"]. Chains to find buried residues in
        output_csv_fname: (str, opt). If present than write csv to this filename otherwise don't write it out

    Returns:
        interf_res: (list of (int, str) tuples

    """
    bsa_values = []  # Here store interface res nums
    # Create Parser objects for each
    apo = DSSPData()
    apo.parseDSSP(apo_dsspf)
    comp = DSSPData()
    comp.parseDSSP(complex_dsspf)

    # Get relative solvent accessibility for the apo chains
    for chain in apo_chains:
        apo_sasas = apo.get_resnumi_aa_sasa_ss_tuples(chain=chain, ignore_cys_pairs=ignore_cys_pairs)
        complex_sasas = comp.get_resnumi_aa_sasa_ss_tuples(chain=chain, ignore_cys_pairs=ignore_cys_pairs)

        # Make sure the residue numbers and amino acid types are the same
        assert([(n, a) for n, a, _, _ in apo_sasas] == [(n, a) for n, a, _, _ in complex_sasas])

        # Loop over the sasa values and store residues with a change in SASA
        for apo_tup, comp_tup in zip(apo_sasas, complex_sasas):
            resn = apo_tup[0]
            aa = apo_tup[1]
            delta_sasa = int(apo_tup[2]) - int(comp_tup[2])
            bsa_values.append((resn, apo_tup[1], delta_sasa))

    return bsa_values


def make_structs_for_bsa_analysis(pdb_id, ag_ch, ab_ch, i):
    """
    Args:
        ag_ch -- (str) Example: "A+B+C+D" or "A"
        ab_ch -- (str) Example: "A+B+C+D" or "A"
    """
    from pymol import cmd
    # First half is making pdb files
    starting_dir = os.getcwd()  # To not affect other programs get the starting dir
    os.chdir("../pdbs/other_ab-ag_complexes/230220_CovAbDab_all")  # To shorten all the file names
    # Only create pdbs if I haven't previously. Separate into ag alone and with ab
    if not os.path.exists("Structs_For_BSA_Analysis/%s_%d_complex.pdb" % (pdb_id, i)) or not os.path.exists("Structs_For_BSA_Analysis/%s_%d_ag.pdb" % (pdb_id, i)):
        # Load full pdb
        if os.path.exists("Downloaded_Structures/%s_imgt.pdb" % pdb_id):
            cmd.load("Downloaded_Structures/%s_imgt.pdb" % pdb_id, object="full")
        else:
            cmd.fetch(pdb_id, type="pdb", name="full", file="temp.pdb")
            os.remove("temp.pdb")

        cmd.remove("solvent")
        # Save a file with the complex
        cmd.save("Structs_For_BSA_Analysis/%s_%d_complex.pdb" % (pdb_id, i), "chain %s+%s" % (ag_ch, ab_ch))
        # Save another file with just the antigen
        cmd.save("Structs_For_BSA_Analysis/%s_%d_ag.pdb" % (pdb_id, i), "chain " + ag_ch)
        cmd.reinitialize()

        # Second half is making dssp files
    os.chdir(r"C:\Users\clint\Box\IG Lab\Members\Clint Holt\Corona\54043-5\dssps\other_ab-ag_complexes")
    # Create dssp files from the complex and the ag alone if I haven't already
    if not os.path.exists("%s_%d_complex.dssp" % (pdb_id, i)) or not os.path.exists("%s_%d_ag.dssp" % (pdb_id, i)):
        fin_comp = "../../pdbs/other_ab-ag_complexes/230220_CovAbDab_all/Structs_For_BSA_Analysis/%s_%d_complex.pdb" % (pdb_id, i)
        fin_ag = "../../pdbs/other_ab-ag_complexes/230220_CovAbDab_all/Structs_For_BSA_Analysis/%s_%d_ag.pdb" % (pdb_id, i)

        os.system("mkdssp -i %s -o %s" % (fin_comp, "%s_%d_complex.dssp" % (pdb_id, i)))  # Run the dssp program
        os.system("mkdssp -i %s -o %s" % (fin_ag, "%s_%d_ag.dssp" % (pdb_id, i)))

    os.chdir(starting_dir)


def get_all_sabdab_bsas(sab_cov_df):
    # Loop over human abs first
    # Skipped 7l06 and 7l02 go back to that
    all_bsa_values = []
    for i in range(len(sab_cov_df)):
        # Scrape excel data from one row
        x = sab_cov_df.loc[i, "antigen_chain"]
        try:
            if math.isnan(sab_cov_df.loc[i, "antigen_chain"]):  # Skip these
                continue
        except TypeError:
            pass
        try:
            pdb = sab_cov_df.loc[i, "pdb"]
            ab_ch = sab_cov_df.loc[i, "Hchain"] + "+" + sab_cov_df.loc[i, "Lchain"]
            ag_ch = sab_cov_df.loc[i, "antigen_chain"].replace(" | ", "+")
        except TypeError:
            print("TypeError", pdb)
            continue
        if pdb in ["7l06", "7l02"]:
            continue

        # Make 2 pdbs and 2 dssps (ag only and bound to ab)
        try:
            make_structs_for_bsa_analysis(pdb, ag_ch, ab_ch, i)
        except FileNotFoundError as fe:
            print(pdb, fe)
            continue
        # From the dssps get a list of tuples with AA, resnum, bsa
        print(pdb, ag_ch)
        bsa_values = get_buried_res(apo_dsspf="../../dssps/other_ab-ag_complexes/%s_ag.dssp" % pdb, complex_dsspf="../../dssps/other_ab-ag_complexes/%s_complex.dssp" % pdb,
                                              apo_chains=ag_ch.split("+"))
        all_bsa_values.append((pdb + "_" + str(i), bsa_values))  # And store all of it in a variable
    return all_bsa_values

def get_ep_res(bsa_list):
    return [(resn, aa, sasa) for resn, aa, sasa in bsa_list if int(sasa) > 0]


def get_s2_ep_res(bsa_list):
    s2_ep = []
    for resn, aa, sasa in bsa_list:
        if int(sasa) > 0 and int(resn) > 685:  # > 685
            s2_ep.append((int(resn), int(sasa)))
    return s2_ep


def get_fasta_tuples(fasta_file):
    """ Convert a given fasta file to tuples of (header, sequence).

        Args:
            fasta_file: str`
                a file name for a FASTA file with '>' preceding each entry.
        """
    sequence_tuples = []
    with open(fasta_file) as light_f:
        for line in light_f.readlines():
            line = line.replace('\n', '')
            try:
                if line[0] == '>':
                    sequence_tuples.append(tuple(current_entry))
                    current_entry = [line, '']
                else:
                    # If this raises an exception then the fasta file doesn't begin with a '>'
                    current_entry[1] += line
            # This catches the first loop through when current_entry doesn't exist
            except NameError:
                current_entry = [line, '']
            except IndexError:  # Some files have empty lines and don't have index 0
                pass

    sequence_tuples.append(tuple(current_entry))  # Catches the last entry
    return sequence_tuples
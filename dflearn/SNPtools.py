import numpy as np
import pandas as pd


def read_sas_ptid(filepath, ic_ptid = 0):
    '''
    Read a sas file, using a column number as FID, with IID = 1 for index [FID, IID]
    
    Parameters
    ----------
    filepath : str
        file path and name
    ic_ptid : int
        column number for FID
    
    Returns
    -------
    op : DataFrame
    '''
    op = pd.read_sas(filepath)
    ptid = op.columns[ic_ptid]
    op = op.drop_duplicates(ptid).assign(FID = lambda x: x[ptid].astype("int64"), IID = 1).set_index(['FID', 'IID']).drop([ptid], axis = 1)
    return(op)


def read_plink(filepath, na = -1):
    '''
    Read a PLINK binary BED/BIM/FAM file, missing SNP values are denoted by -1
    
    Parameters
    ----------
    filepath : str
        file path and name
    
    Returns
    -------
    op : DataFrame
    '''
    geno_recode = {1:na,  # Unknown genotype
                   2: 1,  # Heterozygous genotype
                   0: 2,  # Homozygous A1
                   3: 0}  # Homozygous A2
    geno_values = np.array([[geno_recode[(i >> j) & 3] for j in range(0, 7, 2)] for i in range(256)], dtype=np.int8)
    daa_bim = pd.read_csv("{}.bim".format(filepath), delim_whitespace = True, header = None)
    daa_fam = pd.read_csv("{}.fam".format(filepath), delim_whitespace = True, header = None)
    da0 = np.fromfile("{}.bed".format(filepath), dtype=np.uint8)
    if(np.all(da0[:3] == [108, 27, 1]) & (da0[3:].shape[0] == (daa_bim.shape[0]*np.ceil(daa_fam.shape[0]/4)))):
        da0 = np.reshape(da0[3:], (daa_bim.shape[0], int(np.ceil(daa_fam.shape[0]/4))))
        op = pd.DataFrame(np.reshape(geno_values[da0], (da0.shape[0], da0.shape[1]*4))[:,:daa_fam.shape[0]], index = daa_bim.iloc[:,1].tolist(), columns = pd.MultiIndex.from_arrays(daa_fam.iloc[:, [0, 1]].values.T, names = ["FID", "IID"])).T
    return(op)


def write_plink(df, df_bim, filepath, na = -1):
    '''
    Write genotypes to binary file.
    
    Parameters
    ----------
    df : DataFrame
        the genotypes to write in the BED file
    '''
    daa_fam = pd.DataFrame(df.index.values)
    if daa_fam.shape[1] == 1:
        daa_fam["IID"] = 1
    daa_fam["fatherIID"] = 0
    daa_fam["motherIID"] = 0
    daa_fam["sex"] = 0
    daa_fam["pheno"] = 0
    daa_fam.to_csv("{}.fam".format(filepath), sep=" ", index=False, header=False)
    df_bim.iloc[:, :6].to_csv("{}.bim".format(filepath), sep = "\t", index=False, header=False)
    
    byte_recode = {na:1,  # Unknown genotype
                   1: 2,  # Heterozygous genotype
                   2: 0,  # Homozygous A1
                   0: 3}  # Homozygous A2
    geno_array = df.values.copy()

    for i, i_val in byte_recode.items():
        geno_array[df.values==i] = i_val

    geno_array = np.pad(geno_array, [[0, 4 - len(df)%4 - 4*(len(df)%4==0)], [0, 0]], "constant", constant_values=0).T.astype("uint8")
    geno_byte = geno_array[:, range(0, geno_array.shape[1], 4)].copy()
    
    for i in range(1, 4):
        geno_byte |= geno_array[:, range(i, geno_array.shape[1], 4)] << 2*i
        
    with open("{}.bed".format(filepath), "wb") as file:
        file.write(np.insert(geno_byte.flatten(), 0, [108, 27, 1]))


def read_dosage(filepath):
    '''
    Read a dosage INFO/DOSE file
    
    Parameters
    ----------
    filepath : str
        file path and name
    
    Returns
    -------
    daa : DataFrame
        info file
    da : DataFrame
        dosage file
    '''
    daa = pd.read_csv("{}.info".format(filepath), sep = "\t", index_col = 0)
    da = pd.read_csv("{}.dose".format(filepath), delim_whitespace = True, index_col = 0, header = None, names = ["ptID", "CASE"] + daa.index.tolist()).drop("CASE", axis = 1).drop_duplicates()
    return(daa, da)


def read_ref_dosage(file_ref, file_dosage, fclbeta = lambda x: x["Effect"]):
    daw = pd.read_excel(file_ref).dropna(subset = ["Allele"]).assign(beta = fclbeta)
    daw["Allele"] = daw["Allele"].str[0]
    daw["SNP"] = daw["SNP"].str.strip()
    daa = pd.read_csv("{}.info".format(file_dosage), sep = "\t", index_col = 0)
    da = pd.read_csv("{}.dose".format(file_dosage), delim_whitespace = True, header = None, dtype = {"ptID": str}, names = ["ptID", "CASE"] + daa.index.tolist()).drop_duplicates(subset = ["ptID"])
    da = da.assign(FID = da["ptID"].str[:12].astype("int64"), IID = 1).set_index(["FID", "IID"])
    daw = daw.drop_duplicates(["SNP"]).set_index("SNP")[['beta', "Allele"]].join(daa["ALT(1)"], how = 'inner')
    daw["beta_adjusted"] = (daw["beta"]*(2*(daw["Allele"] == daw["ALT(1)"]) - 1)*(~daw["Allele"].isnull())).fillna(0)
    return(daw, da.loc[:,daw.index])

def RiskScore(X, W):
    '''
    Calculate risk score given weights, with missing values filled with mean
    
    Parameters
    ----------
    X : DataFrame
        data to calculate risk score
    W : DataFrame
        weights of variables

    Returns
    -------
    op : DataFrame
        risk score
    '''
    op = X.fillna(X.mean()).dot(W).add(-2*W.where(W < 0, 0).sum())
    return(op)


def vcf_to_RFMix(file_vcf, file_map, file_out):
    '''
    Covert vcf and hapmap genetic location (cM) files to alleles and snp_locations file for RFMix
    
    Parameters
    ----------
    file_vcf : str
        vcf file of location and subjects of SNPs
    file_map : str
        genetic location file of hapmap
    file_out : str
        name of output file, with "_alleles.txt" and "_snp_locations.txt" appended

    Notes
    -----
    file_out + "_alleles.txt" : file
        alleles data used for RFMix
    file_out + "_snp_locations.txt" : file, snp locations (centiMorgan) data used for RFMix
    '''
    print("Reading vcf file")
    da = pd.read_csv(file_vcf, delim_whitespace=True, skiprows = 5)
    daa = pd.read_csv(file_map, sep = "\t")
    X = da.iloc[:, 9:]
    print("Read vcf file: {1} subjects, {0} SNPs".format(*X.shape))
    op_cM = np.interp(da["POS"], daa["Position(bp)"], daa["Map(cM)"], left = 0, right = daa["Map(cM)"].max())
    np.savetxt("{}_snp_locations.txt".format(file_out), op_cM)
    print("Interpolate gene location on {} SNPs, min: {:.4f} cM, max: {:.4f} cM".format(len(op_cM), op_cM.min(), op_cM.max()))
    split = lambda s: [s.str[0], s.str[2]]
    op_X = pd.concat(sum([split(X.iloc[:,i]) for i in range(X.shape[1])], []), axis = 1).values.astype("int8")
    np.savetxt("{}_alleles.txt".format(file_out), op_X, fmt = "%i", delimiter = "")
    print("Finished for {0} SNPs and {1} phase sequences \nOutput files: {2}_snp_locations.txt, {2}_alleles.txt".format(op_X.shape[0], op_X.shape[1], file_out))
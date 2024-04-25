def parallel_searchlight(neigh):

    # create dataset
    dataset = rsa.data.Dataset(
        data_2d[:, neigh],
        channel_descriptors={'channel': np.array(['center_' + str(x) for x in range(len(neigh))])},
        obs_descriptors={'cue': cue, 'blocks': blocks})

    # euclidean distance
    rdm_eucl = rsa.rdm.calc_rdm_unbalanced(dataset, method='euclidean', descriptor='cue')

    # mahalanobis
    # noise_prec_diag = rsa.data.noise.prec_from_unbalanced(dataset, obs_desc='cue', method='diag')
    # noise_prec_shrink_eye = rsa.data.noise.prec_from_unbalanced(dataset, obs_desc='cue', method='shrinkage_eye')
    noise_prec_shrink_diag = rsa.data.noise.prec_from_unbalanced(dataset, obs_desc='cue', method='shrinkage_diag')
    rdm_maha = rsa.rdm.calc_rdm_unbalanced(dataset, method='mahalanobis', descriptor='cue', noise=noise_prec_shrink_diag)

    return rdm_eucl, rdm_maha


participant_id = 'subj100'
experiment = 'smp1'
glm = '1'
path = os.path.join(gl.baseDir, experiment, gl.glmDir + glm, participant_id)

reginfo = pd.read_csv("{}/{}_reginfo.tsv".format(path, participant_id), sep='\t')
reginfo.drop(reginfo[reginfo['cue'] == 'rest'].index, inplace=True)

cue = reginfo.cue.to_list()

tmp_img = nb.load(path + '/beta_0001.nii')
mask = ~np.isnan(tmp_img.get_fdata())
x, y, z = tmp_img.get_fdata().shape

flattened_data = list()
data = np.zeros((len(reginfo), x, y, z))
blocks = np.zeros(len(reginfo))
for i, im in enumerate(reginfo.iterrows()):
    print('processing: beta_%04d.nii' % (im[0] + 1))
    data[i] = nb.load(path + '/beta_%04d.nii' % (im[0] + 1)).get_fdata()
    # data_array = data.get_fdata().ravel()
    # flattened_data.append(data_array[mask_array])
    blocks[i] = im[1].run

centers, neighbors = get_volume_searchlight(mask, radius=5, threshold=1)

data_2d = data.reshape([data.shape[0], -1])
data_2d = np.nan_to_num(data_2d)

results = Parallel(n_jobs=4, verbose=10)(delayed(parallel_searchlight)(neigh) for neigh in neighbors)

# Extract results
rdm_eucl, rdm_maha = zip(*results)

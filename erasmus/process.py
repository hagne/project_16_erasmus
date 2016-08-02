import os
import atmPy


def process_peak(file_name,
                 file_folder_name,
                 cal,
                 time_shift = False,
                 d_start = 130,
                 d_end = 3000,
                 no_bin = 30,
                 sampling_eff = False,
                 norm2flowrate = True,
                 corr_battery_fail = False,
                 verbose = False):

    if verbose:
        print(('--------------\n' 'process_peak: %s %s\n'%( file_folder_name, file_name)))

    fname_peak = file_folder_name + file_name
    m = peaks.read_binary(fname_peak, time_shift =time_shift)
    m.apply_calibration(cal)
    bins = np.logspace(np.log10(d_start), np.log10(d_end), no_bin)
    dist = m.peak2sizedistribution(bins)
    if norm2flowrate:
        if verbose:
            print('\t - normalizing to flow rate')
        dist /= 3.  # normalizing to flow rate
        dist.norm2flowrate = True
    else:
        dist.norm2flowrate = False
    if sampling_eff:
        if 'Flight' in file_folder_name:
            if verbose:
                print('\t - correcting to sampling efficiency')
            sampleff = sampling_efficiency_pilatus(d = dist.bincenters/1000.)  # todo (speedup): no need to calculate that each time stays the same anyway.
            dist.data = dist.data.rmul(sampleff["sampling_eff."].values)
            dist.sampling_eff = True
        else:
            dist.sampling_eff = False
    else:
        dist.sampling_eff = False

    dist.corr_battery_fail = False
    if corr_battery_fail:
        ts = dist.get_timespan()[0]
        if ts.year == 1970:
            if verbose:
                print('\t - correcting time error due to battery fail')
            fts = file_folder_name.split('/')[-2]
            fts = pd.to_datetime(fts,format= '%Y%m%d_%H%M')
            dt = fts - ts
            dist.data.index += dt
            dist.corr_battery_fail = True

    return dist


def save2netcdf(dist, folder_nc, fol, file, version):
    unit_time = 'days since 1900-01-01'

    # file name
    fname_nc = folder_nc + fol + '__' + '_'.join(file.split('_')[:2]) + '.nc'

    # write the file
    file_mode = 'w'
    try:
        ni = Dataset(fname_nc, file_mode)
    except RuntimeError:
        if os.path.isfile(fname_nc):
            os.remove(fname_nc)
            ni = _Dataset(fname_nc, file_mode)

    # dimensions
    ni.createDimension('time', dist.data.shape[0])
    ni.createDimension('bins', dist.data.shape[1])
    ni.createDimension('bin_edges', dist.bins.shape[0])

    # time variable
    ts_time_num = date2num(dist.data.index.to_pydatetime(), unit_time)
    time_var = ni.createVariable('time', ts_time_num.dtype, 'time')
    time_var[:] = ts_time_num
    time_var.units = 'days since 1900-01-01'
    time_var.data_period = '1 second'
    time_var.timestampposition = 'beginning of averaging period'

    # size dist bin edges variable
    # ts_columns = ts.data.columns.values.astype(str)
    var_bin_edges = ni.createVariable('bin_edges', dist.bins.dtype, 'bin_edges')
    var_bin_edges[:] = dist.bins
    var_bin_edges.units = 'nm'

    # size dist data varible
    var_data = ni.createVariable('sizedistribution', dist.data.values.dtype, ('time', 'bins'))
    var_data[:] = dist.data.values
    var_data.units = dist.distributionType

    # particle number,surface,volume concentration variable
    var_num_conc = ni.createVariable('particle_number_concentration', dist.particle_number_concentration.data.values.dtype, 'time')
    var_num_conc[:] = dist.particle_number_concentration.data.values
    var_num_conc.units = "#/cm^3"
    #
    var_surf_conc = ni.createVariable('particle_surface_concentration', dist.particle_surface_concentration.data.values.dtype, 'time')
    var_surf_conc[:] = dist.particle_surface_concentration.data.values
    var_surf_conc.units = 'um^2/cm^3'
    #
    var_vol_conc = ni.createVariable('particle_volume_concentration', dist.particle_volume_concentration.data.values.dtype, 'time')
    var_vol_conc[:] = dist.particle_volume_concentration.data.values
    var_vol_conc.units = 'um^3/cm^3'

    # particle concentrations for particles larger than the detection range
    var_num_toobig = ni.createVariable('particle_number_concentration_outside_range',
                                       dist.particle_number_concentration_outside_range.data.values.dtype,
                                       'time')
    var_num_toobig[:] = dist.particle_number_concentration_outside_range.data.values
    var_num_toobig.units = "#/cm^3"

    #########
    ## Attribute

    ni.setncattr('location', 'Oliktok point, AK')
    ni.setncattr('instrument', 'POPS')
    ni.setncattr('campain', 'Erasmus spring 2015')
    ni.setncattr('plattform', 'Pilatus UAS')
    ni.setncattr('retrival_version', version)
    ni.close()

    return fname_nc

def sampling_efficiency_pilatus(d=np.logspace(np.log10(0.15), np.log10(2.5), 500),
                                sampling_flow_rate=3,
                                ambient_air_speed=25.5,
                                inlet_length=1.651,  # m ... 14 + 33 + 7 + 11 inches
                                inlet_diameter=0.0015875,
                                temperature=293.15):


    inlet_eff = se.inlet_efficiency_isoaxial_horizontal_sharp_edged(temperature=temperature,
                                                                    #                                                     pressure = 101.3,
                                                                    particle_diameter=d,
                                                                    #                                                     particle_density=1000,
                                                                    inlet_diameter=inlet_diameter,
                                                                    inlet_length=inlet_length,
                                                                    sampling_flow_rate=sampling_flow_rate,
                                                                    air_velocity_inlet=False,
                                                                    ambient_air_speed=ambient_air_speed,
                                                                    velocity_ratio=False,
                                                                    verbose=False)

    bent_eff = se.loss_in_a_bent_section_of_circular_tubing(temperature=temperature,
                                                            #                                              pressure=101.3,
                                                            particle_diameter=d,
                                                            #                                              particle_density=1000,
                                                            tube_air_flow_rate=sampling_flow_rate,
                                                            tube_air_velocity=False,
                                                            tube_diameter=inlet_diameter,
                                                            angle_of_bend=90,
                                                            flow_type='auto',
                                                            verbose=False)

    total_eff = inlet_eff * bent_eff ** 5

    df = pd.DataFrame(np.array([total_eff]).transpose(), index=d * 1000,
                      columns=['sampling_eff.'])
    df.index.name = 'diameter_nm'
    return df

def process_campain(version,
                    change_log,
                    base_folder='/Users/htelg/data/2016_04_olikta_point/',
                    fname_cal='/Users/htelg/data/POPS_calibrations/150317_Pilatus_DOS.csv',
                    time_shift=False,
                    sampling_eff=False,
                    norm2flowrate=True,
                    corr_battery_fail=True,
                    overwrite=False,
                    verbose=False,
                    test=False
                    ):


    """
    Parameters
    ----------
    change_log: list"""
    cal = calibration.read_csv(fname_cal)
    folders = os.listdir(base_folder)
    if verbose:
        print(folders)
    # generate out put folder
    timestamp = time_tools.get_timestamp().replace('-', '').replace(':', '').replace(' ', '_')
    folder_nc = base_folder + 'processed_' + timestamp
    if test:
        folder_nc += '_test'
        test_success = False
    folder_nc += '/'

    if os.path.isdir(folder_nc):
        if not overwrite:
            txt = 'A directory with the name %s already exist. Set overwrite to True if you want to overwrite existing data'
            raise ValueError(txt)

    os.mkdir(folder_nc)
    if verbose:
        print('output folder name: ', folder_nc)

    ##################
    # write readme
    readme = open(folder_nc + 'README', 'w')
    readme.write('retrival version: %s\n' % version)
    readme.write('======================\n')

    readme.write('Applied corrections etc.\n')
    readme.write('------------------------\n')
    readme.write('\t - normalized to avg. flow rate of 3cc/s\n')
    readme.write('\t - corrected for sampling efficiency for assumed air-speed of 25.5 m/s\n')
    if time_shift:
        readme.write('\t - applied time shift: %s\n' % (str(time_shift)))

    # write change log
    readme.write('\nChange log:\n==========\n')
    readme.write('\n'.join(change_log) + '\n')

    # file by file
    readme.write('Perfomed data operations\n')
    readme.write('------------------------\n')
    #############
    # do the loop
    results = []
    for fol in folders:
        if test:
            if fol != test:
                if 0:
                    print('%s not equal %s' % (fol, test))
                continue
                #     print(base_folder + fol)
        fol_path = base_folder + fol
        try:
            files = os.listdir(fol_path)
        except NotADirectoryError:
            continue
        date = fol.split('_')[0]
        try:
            ts = pd.Timestamp(date)
        except ValueError:
            continue

        if verbose:
            print('In folder: %s' % fol)

        no_of_bin_files = 0
        dist_list = []
        for file in files:
            if '_Peak.bin' in file:
                if no_of_bin_files == 0:
                    readme.write('%s \n' % fol)
                if verbose:
                    print('\t processing file: %s' % file)
                file_folder = base_folder + fol + '/'
                dist = process_peak(file, file_folder, cal,
                                    time_shift=time_shift,
                                    sampling_eff=sampling_eff,
                                    norm2flowrate=norm2flowrate,
                                    corr_battery_fail=corr_battery_fail,
                                    verbose=verbose)
                dist_list.append(dist)
                #                 fname_nc = save2netcdf(dist, folder_nc, fol, file, version)
                #                 result = {}
                #                 result['sizedistribution'] = dist
                #                 result['fname_output'] = fname_nc
                #                 result['fname_input'] = file_folder
                #                 results.append(result)

                # log in readme
                readme.write('\t %s \t' % file)
                operations = []
                if dist.norm2flowrate:
                    operations.append('flow rate')
                if dist.sampling_eff:
                    operations.append('sampling eff.')
                readme.write(', '.join(operations) + '\n')
                #                 break
        if test:
            test_success = True
        ###########
        # concatenate multiple dists
        dist = dist_list[0]
        if len(dist_list) > 1:
            for sd in dist_list[1:]:
                ts = dist.get_timespan()
                td = ts[1] - ts[0] + pd.Timedelta(1, 's')
                sd.data.index += td
                dist.data = pd.concat([dist.data, sd.data])
                dist.particle_number_concentration_outside_range.data = pd.concat([dist.particle_number_concentration_outside_range.data, sd.particle_number_concentration_outside_range.data])

        fname_nc = save2netcdf(dist, folder_nc, fol, file, version)

        result = {}
        result['sizedistribution'] = dist
        result['fname_output'] = fname_nc
        result['fname_input'] = file_folder
        results.append(result)

    readme.close()
    print('done')
    if test:
        if not test_success:
            print('test faild')
        return results
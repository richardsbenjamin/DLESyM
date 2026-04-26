import h5py
import dask.array as da
import xarray as xr
import pandas as pd
import cftime
import time
import pickle
import os
import logging

def to_cftime(dt):
    return cftime.DatetimeGregorian(dt.astype(object).year, dt.astype(object).month, dt.astype(object).day, dt.astype(object).hour, dt.astype(object).minute)

def h5py_to_da(f: 'file', start_idx: int, end_idx: int, type_: str, args: "Namespace", pred_shape: tuple):
    return da.from_array(f[f'{type_}_prediction'], 
                                        chunks=(args.time_chunk, 
                                                args.step_chunk, 
                                                pred_shape[2],
                                                pred_shape[3],
                                                pred_shape[4],
                                                pred_shape[5],))

def get_datetime_steps(ocean_start, ocean_leads, atmos_start, atmos_leads, forecast_dates):
    ocean_steps = [to_cftime(lt) for lt in [ocean_start] + list(ocean_leads)]
    atmos_steps = [to_cftime(lt) for lt in [atmos_start] + list(atmos_leads)]
    formatted_dates = [to_cftime(f.astype('datetime64[m]')) for f in forecast_dates]
    
    return ocean_steps, atmos_steps, formatted_dates

def get_timedelta_steps(ocean_leads, atmos_leads):
    ocean_steps = [pd.Timedelta(hours=0)] + list(ocean_leads)
    atmos_steps = [pd.Timedelta(hours=0)] + list(atmos_leads)
    
    return ocean_steps, atmos_steps

def to_dataarray(
    data, 
    forecast_dates, 
    steps, 
    cfg, 
    meta_ds,
):
    channels = cfg.data.output_variables or cfg.data.input_variables
    return xr.DataArray(
        data,
        dims=['time', 'step', 'channel_out', 'face', 'height', 'width'],
        coords={
            'time': forecast_dates,
            'step': steps,
            'channel_out': channels,
            'face': meta_ds.face,
            'height': meta_ds.height,
            'width': meta_ds.width
        },
    )

def rescale_and_convert(prediction_da, data_module, cfg):
    scaling = data_module.test_dataset.target_scaling
    prediction_da[:] *= scaling['std']
    prediction_da[:] += scaling['mean']
    ds = prediction_da.to_dataset(dim='channel_out')
    for variable in ds.data_vars:
        var_scaling_cfg = cfg.data.scaling.get(variable, {})
        epsilon = var_scaling_cfg.get('log_epsilon')
        
        if epsilon is not None:
            ds[variable] = np.exp(
                ds[variable] + np.log(epsilon)
            ) - epsilon
            
    return ds

def to_chunked_dataset(ds, chunking):
    """
    Create a chunked copy of a Dataset with proper encoding for netCDF export.
    :param ds: xarray.Dataset
    :param chunking: dict: chunking dictionary as passed to
        xarray.Dataset.chunk()
    :return: xarray.Dataset: chunked copy of ds with proper encoding
    """
    chunk_dict = dict(ds.dims)
    chunk_dict.update(chunking)
    for var in ds.data_vars:
        if 'coordinates' in ds[var].encoding:
            del ds[var].encoding['coordinates']
        ds[var].encoding['contiguous'] = False
        ds[var].encoding['original_shape'] = ds[var].shape
        ds[var].encoding['chunksizes'] = tuple([chunk_dict[d] for d in ds[var].dims])
        ds[var].encoding['chunks'] = tuple([chunk_dict[d] for d in ds[var].dims])
    return ds


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

logger = logging.getLogger(__name__)
logging.getLogger('cfgrib').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)


if __name__ == "__main__":
    

    with open(f"{cache_dir}/args.pkl", "rb") as f:
        namespace = Namespace(**pickle.load(f))

    cluster = LocalCluster(n_workers=2, threads_per_worker=2, memory_limit='16GB')
    client = Client(cluster)

    types = ['ocean', 'atmos']

    for type_ in types:

        f = h5py.File(f'{namespace.args.cache_dir}/{type_}_prediction.hdf5', 'r')
        shape = f[f'{type_}_prediction'].shape
        total_timesteps = shape[1]

        batch_size = 50

        logger.info(f"Total timesteps: {total_timesteps}. Processing in batches of {batch_size}.")

        for start_idx in range(0, total_timesteps, batch_size):
            end_idx = min(start_idx + batch_size, total_timesteps)
            print(f"Processing batch {start_idx} to {end_idx}...")
            
            batch_array = h5py_to_da(
                f, start_idx, end_idx, type_, namespace.args, shape)

            batch_da = to_dataarray(
                batch_array,
                forecast_dates[start_idx:end_idx], 
                getattr(namespace, f"{type_}_steps"),
                getattr(namespace, f"{type_}_cfg"),
                getattr(namespace, f"{type_}_meta_ds"),
            )
            batch_ds = rescale_and_convert(
                batch_da,
                getattr(namespace, f"{type_}_data_module"),
                getattr(namespace, f"{type_}_cfg")
            )

            if start_idx == 0:
                batch_ds.to_zarr(gcs_ocean_path, mode="w", compute=True)
            else:
                batch_ds.to_zarr(gcs_ocean_path, append_dim='time', compute=True)

        f.close()
        logger.info(f"Finished transfer of {type_} data.")


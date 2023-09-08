import numpy as np
import dask


def function_3(n):
    a = np.linspace(n,n+100)
    return a


@dask.delayed
def function_2(n):
    a = np.log(function_3(1+n**2)) / np.sqrt(n)
    b = np.log(a) * (-1)**n
    r = np.random.uniform()
    return (a + b) * r**np.sqrt(np.sqrt(n)) - np.log(r/n)


def function_1(n):
    list_of_delayed_tasks = []
    for i in range(1,n+1):
        delayed_task = function_2(i)
        list_of_delayed_tasks.append(delayed_task)
    return list_of_delayed_tasks


def random_numbercrunching_in_parallel(n=5000):
    from_compute = dask.compute(function_1(n))
    result = np.nanmean(np.array(from_compute[0]), axis=1)
    val = np.nanmean(result)
    print('Some numbercrunching on the cluster resulted in the value', val)
    return val


@dask.delayed
def animate(frames):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import matplotlib
    import cartopy.crs as ccrs
    matplotlib.rcParams['animation.embed_limit'] = 2**128
    
    array_3d, extent, times, data_name = get_timeseries()

    # Create a figure and axis
    projection, data_crs = ccrs.PlateCarree(), ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 1, figsize=(15, 5), subplot_kw={'projection': projection})
    cax = fig.add_axes([0.71, 0.275, 0.007, 0.39])  # Add an axis for the colorbar

    # Create the initial plot for the first frame
    pvt, vmin, vmax = np.nanmean(array_3d), np.nanmin(array_3d) + 7, np.nanmax(array_3d) - 7
    im = ax.imshow(array_3d[0, :, :], extent=extent,origin='upper', transform=data_crs, vmin=vmin, vmax=vmax)

    # Add latitude and longitude gridlines
    gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=0, color='w', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(' Kelvin ')  # Add the colorbar label here

    def update(frame):
        im.set_data(array_3d[frame, :, :])
        im.set_clim(vmin=vmin, vmax=vmax)
        ax.set_title(f' {data_name}, timestep :{times[frame]}')      # Update the plot title for the current frame
        return [im]
    
    animation = FuncAnimation(fig, update, frames=frames, blit=True, interval=150)
    plt.close(fig)
    
    return animation


def get_timeseries():
    
    import xarray as xr
    t = xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib")
    array_3d = t.t2m.values
    lat, lon = t.latitude.values, t.longitude.values
    extent = [lon[0],lon[-1],lat[-1],lat[0]]
    times = t.time.values.astype("<M8[m]")
    data_name = t.t2m.attrs["long_name"]
    
    return array_3d, extent, times, data_name


def animate_tutorial_dataset(hours=24):
    from IPython.display import HTML
    animation = dask.compute(animate(hours))
    print('A worker is preparing a short animation generated w/ xarray.tutorial.open_dataset("era5-2mt-2019-03-uk.grib")...')
    show_html = HTML(animation[0].to_jshtml())
    return show_html
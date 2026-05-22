import numpy as np
from shapely.geometry import LineString
import pyproj
import rasterio
import logging as LOGGER
import matplotlib.pyplot as plt
from functools import cached_property


# LOGGER.basicConfig(level=LOGGER.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Push logs to external file
LOGGER.basicConfig(filename='../logs/terrain_loader_output.log', level=LOGGER.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# Pre-selected route PIs (Longitude, Latitude)
PI_COORDS = [
    (81.487570, 28.899105),         # PI0: Upper Karnali site
    (81.256220, 28.831518),         # PI1
    (81.256220, 28.829324),         # PI2
    (81.189072, 28.801362),         # PI3
    (81.189072, 28.801362)          # PI4: Dododhara 400kV subsation
]

class LoadTerrain:
    def __init__(self, pi_coords=PI_COORDS):
        self.pi_coords = pi_coords
        self.wgs84 = pyproj.CRS("EPSG:4326")
        self.utm44n = pyproj.CRS("EPSG:32644")       # UTM Zone 44N for Western Nepal
        self.metric_pis = self.coords_projection_wgs84_to_utm44n()
        self.route_line = self.set_route_line()

    def coords_projection_wgs84_to_utm44n(self):
        """Projects PI points (WGS884 Lat/Lon) to metric Easting/Northing coordinates (UTM Zone 44N meters)"""    
        LOGGER.info("Projecting PI points (WGS884 Lat/Lon) to metric Easting/Northing coordinates (UTM Zone 44N meters)")    
        projector = pyproj.Transformer.from_crs(self.wgs84, self.utm44n, always_xy=True).transform
        return [projector(lon, lat) for lon, lat in self.pi_coords]

    def set_route_line(self):
        """Connect metric-pis with linestring"""
        return LineString(self.metric_pis)

    @cached_property
    def compute_tls_length(self):
        """Returns the total transmission line length (km)"""
        return self.route_line.length / 1000

    def generate_x_z_profile(self):
        """Process DEM and generate a detailed data point every 20 meters along transmission line path
        
        profile_x represents the distance from PI0 to PI4
        profile_z represents the elevation values in meters"""
        LOGGER.info("Generating x and z dimension profiles from DEM data")    
        sampling_interval = 20
        distances = np.arange(0, self.route_line.length, sampling_interval)
        geo_projector = pyproj.Transformer.from_crs(self.utm44n, self.wgs84, always_xy=True).transform

        profile_x = []
        profile_z = []

        with rasterio.open("../nepal-NASADEM/output_nepal.tif") as dem:
            for d in distances:
                point_on_line = self.route_line.interpolate(d)
                lon, lat = geo_projector(point_on_line.x, point_on_line.y)
                try:
                    val = list(dem.sample([(lon, lat)]))[0][0]
                    # Filter out any nodata anomalies
                    if val > -100 and val < 9000: 
                        profile_x.append(d)
                        profile_z.append(val)
                except IndexError:
                    LOGGER.error("Path strayed outside of DEM frame boundaries")
                    continue
        profile_x = np.array(profile_x)
        profile_z = np.array(profile_z)
        LOGGER.info(f"Generation complete")
        LOGGER.info(f"profile_x: {profile_x}\nprofile_z: {profile_z}")
        return profile_x, profile_z

    def visualize_terrain(self, export_img=True, export_csv=True):
        IMG_PATH = "../output/terrain_profile.png"
        DATA_PATH = "../data/route_profile.csv"
        LOGGER.info(f"Visualizing terrain. Export Image set to {export_img}. Export csv set to {export_csv}")
        profile_x, profile_z = self.generate_x_z_profile()
        plt.figure(figsize=(14, 5))
        plt.plot(profile_x / 1000, profile_z, color="#2c3e50", linewidth=1.5, label="Ground Terrain")

        plt.title(f"Longitudinal Terrain Profile: Upper Karnali to 400kV Interconnection Indian Border ({self.compute_tls_length:.2f} km)", fontsize=12, fontweight='bold')
        plt.xlabel("Cumulative Distance along Route (km)", fontsize=10)
        plt.ylabel("Elevation above Sea Level (m)", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.fill_between(profile_x / 1000, profile_z, color="#ecf0f1", alpha=0.5)
        plt.legend()

        if export_img:
            LOGGER.info(f"Saving plot to {IMG_PATH}")
            plt.savefig(IMG_PATH, dpi=300, bbox_inches='tight')
            plt.show()

        if export_csv:
            LOGGER.info(f"Saving csv to {DATA_PATH}")
            np.savetxt(DATA_PATH, np.column_stack((profile_x, profile_z)), delimiter=",", header="distance_m,elevation_m", comments="")




if __name__ == '__main__':
    terrain = LoadTerrain()
    LOGGER.info(f"Total Transmission Line Route Length: {terrain.compute_tls_length:.2f} km")
    terrain.visualize_terrain()
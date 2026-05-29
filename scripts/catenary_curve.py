import numpy as np
import logging as LOGGER
from functools import cached_property
import matplotlib.pyplot as plt


LOGGER.basicConfig(level=LOGGER.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Push logs to external file
# LOGGER.basicConfig(filename='../logs/catenary_curve_output.log', level=LOGGER.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')



ICE_DENSITY = 915       # kg/m^3
G = 9.81                # m/s


# ACSR MOOSE Conductor Properties
E = 69.0e9          # N/m^2
A = 5.97e-4         # m^2
ALPHA = 19.3e-6     # per C
D = 0.03177         # m
UTS = 159600        # N
WC = 1.998          # kg/m
WN = WC * 9.81      # N/m


# BASELINE CONDITIONS (EDS)
L = 400                 # m
T1 = 32                 # C
WIND_SPEED_1 = 0        # m/s
ICE_1 = 0               # m
EDS_H = 20              # tension % of limit

X = np.linspace(0, L, 200)      # scale for plot

# EXTREMES
T_MIN = -3.87                   # C
T_MAX = 40.27                   # C
WS_MAX = 8.71                   # m/s
PRECIT_MAX = 246.59             # mm


class CatenaryCurve:
    def __init__(self,
                 t2,
                 wind_speed2,
                 ice2,
                 t1=T1,
                 wind_speed1=WIND_SPEED_1,
                 ice1=ICE_1,
                 eds_h=EDS_H):
        """
        Parameters:
        - t1, t2: Initial and final temperatures (°C)
        - wind_speed1, wind_speed2: Wind speed (m/s)
        - ice1, ice2: Radial ice thickness (meters)
        - eds_h: every day stress tension (expressed as % of limit)
        """
        self.t1 = t1
        self.t2 = t2
        self.wind_speed1 = wind_speed1
        self.wind_speed2 = wind_speed2
        self.ice1 = ice1
        self.ice2 = ice2
        self.eds_h = eds_h

    @staticmethod
    def wind_pressure(wind_speed):
        q = 0.6 * (wind_speed ** 2)
        LOGGER.info(f"Computed wind pressure (q) = {q} N/m^2 for {wind_speed} m/s")
        return q

    def resultant_load(self, ice, wind):
        """
        ice : radial ice thickness in m
        wind: wind speed in m/s
        """
        v_load = WN + (np.pi * ice * (D + ice) * ICE_DENSITY * G)
        q = self.wind_pressure(wind_speed=wind)
        h_load = q * (D + 2 * ice)
        wr = np.sqrt(v_load**2 + h_load**2)
        LOGGER.info(f"Computed resultant load (wr) = {wr} N")
        return wr
    
    @cached_property
    def wr1(self):
        """Initial condition resultant load (wr1)"""
        return self.resultant_load(ice=self.ice1, wind=self.wind_speed1)
    
    @cached_property
    def wr2(self):
        """Final condition resultant load (wr2)"""
        return self.resultant_load(ice=self.ice2, wind=self.wind_speed2)
    
    @cached_property
    def h1(self):
        """Initial condition tensionn computed from percentage loading of max tension of conductor. Ex. 20% of UTS"""
        h1 = (self.eds_h / 100) * UTS
        LOGGER.info(f"Computed initial tension (h1) = {h1} N")
        return h1
    
    @staticmethod
    def sag_approximation(wr, span, H):
        sag = (wr * (span**2)) / (8 * H)
        LOGGER.info(f"Computed sag approximation is {sag}")
        return sag

    def solve_tension_polynomial(self):
        LOGGER.info(f"Solving tension by finding cubic polynomial roots")
        C = (self.wr2**2 * (L**2) * E * A) / 24                             # RHS constant
        B = 1                                                               # Coefficient for H2^3
        A_factor = (self.wr1**2 * (L**2) * E * A) / (24 * (self.h1**2))
        thermal_factor = ALPHA * E * A * (self.t2 - self.t1)
        K = A_factor - self.h1 + thermal_factor                                  # Coefficient for H2^2
        Y = 0                                                               # Coefficient for H2
        coefficients = [B, K, Y, -C]
        roots = np.roots(coefficients)
        real_positive_roots = [r.real for r in roots if np.isreal(r) and r > 0]
        if not real_positive_roots:
            raise ValueError("No physical tension solution found. Check input parameters.")
        H2 = real_positive_roots[0]
        diff = self.h1 - H2
        LOGGER.info(f"Final condition tension (H2) = {H2}")
        LOGGER.info(f"Tension dropped by {diff} N")
        return H2

    @staticmethod
    def catenary_profile(H, wr):
        """Generate catenary profile curves. Returns the veritcal components of the catenary path"""
        LOGGER.info(f"Computing catenary profile")
        x = X
        z = (H / wr) * (np.cosh(wr * (x - L / 2) / H) - np.cosh(wr * L / (2 * H)))
        return z * (WC)

    def baseline_catenary_profile(self):
        return self.catenary_profile(H=self.h1, wr=self.wr1)
    



def plot_profiles(y_eds, y_cold):
    x = X
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y_eds, 'b-', label=f'Everyday Stringing (32°C, No Wind), Sag = {-min(y_eds):.2f}m')
    ax.plot(x, y_cold, 'g--', label=f'Min Temp.  (15°C, 45m/s), Vert Sag = {-min(y_cold):.2f}m')
    # ax.plot(x, y_thermal, 'r-', label=f'Max Thermal Sag (85°C, No Wind), Sag = {-min(y_thermal):.2f}m')
    ax.set_title('400m Transmission Line Span Conductor Profiles', fontsize=12, fontweight='bold')
    ax.set_xlabel('Span Distance (meters)')
    ax.set_ylabel('Vertical Drop from Attachment Point (meters)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower center')
    plt.tight_layout()
    plt.savefig('conductor_sag_profiles.png', dpi=300)



def estimate_radial_ice(precipitation_mm, wind_speed_ms, temperature_c):
    """
    Estimates radial ice thickness using the Jones empirical model.
    Assumes standard glaze ice conditions.
    """
    rho_ice = 900.0         # Glaze ice density in kg/m³
    V_t = 5.0               # Average terminal velocity of freezing raindrops (m/s)
    if temperature_c <= 0.5 and precipitation_mm > 0:
        P_mass = precipitation_mm          
        dr = (P_mass / (np.pi * rho_ice)) * np.sqrt(1 + (wind_speed_ms / V_t)**2)
        return dr
    return 0.0



if __name__ == "__main__":
    # CASE 1: Coldest, windiest conditions
    LOGGER.info("======= CASE 1: Coldest, windiest conditions =======")
    ice2 = estimate_radial_ice(precipitation_mm=29.85, wind_speed_ms=WS_MAX, temperature_c=T_MIN)      # From NASA POWER Data
    cold_cc = CatenaryCurve(t2=T_MIN, wind_speed2=WS_MAX, ice2=ice2)
    cold_tension = cold_cc.solve_tension_polynomial()
    cold_catenary_profile = cold_cc.catenary_profile(H=cold_cc.h1, wr=cold_cc.wr2)
    baseline_catenary_profile = cold_cc.baseline_catenary_profile()
    plot_profiles(y_eds=baseline_catenary_profile, y_cold=cold_catenary_profile)




    # # # CASE 2: Hottest condition (max. operating temp. of conductor)
    # # hot_cc = CatenaryCurve(t2=85, wind_speed2=0, ice2=0)
    # # hot_tension = hot_cc.solve_tension_polynomial()

    # # # CASE 3: Max. wind swing condition
    # # wind_cc = CatenaryCurve(t2=32, wind_speed2=WS_MAX, ice2=0)
    # # wind_tension = wind_cc.solve_tension_polynomial()

    # # Plots
    # y_eds = catenary_profile(x, H1, wr1, w_c)
    # y_wind = catenary_profile(x, H_wind, wr_wind, w_c)
    # y_thermal = catenary_profile(x, H_thermal, w_c, w_c)


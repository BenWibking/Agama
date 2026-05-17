#!/usr/bin/env python3
"""
Construct a multicomponent self-consistent galaxy model with a magnetized
isothermal gas disk.

This is a small modification of example_self_consistent_model.py.  The stellar disk,
bulge and halo are still represented by action-based DFs, but the gas disk is rebuilt
before each self-consistent iteration from vertical isothermal MHD balance in
the current total potential.  With no magnetic field, this is

    rho_g(R,z) = Sigma_g(R) exp[-(Phi(R,z)-Phi(R,0)) / c_s^2] / Z(R),

where Z(R) normalizes the density to the requested surface density Sigma_g(R).
An optional ordered toroidal field is represented with constant plasma beta

    beta_B = P_gas / P_mag,
    B_phi(R,z) = sqrt(8 pi c_s^2 rho_g(R,z) / beta_B).

Its magnetic pressure changes the vertical gas equilibrium through
c_eff^2 = c_s^2 (1 + 1/beta_B), and its toroidal tension is included when
computing gas rotation velocities.  The gas density is then inserted as a normal
static density component, so its gravity participates in the next potential solve.

All quantities are in the usual Agama units used by data/SCM.ini:
length = kpc, velocity = km/s, mass = Msun.
"""
from __future__ import print_function

import argparse
import os
import sys

import numpy

agama = None

MSUN_CGS = 1.98847e33
KPC_CGS = 3.0856775814913673e21
KM_CGS = 1.0e5

try:
    from ConfigParser import RawConfigParser  # python 2
except ImportError:
    from configparser import RawConfigParser  # python 3


def import_agama():
    global agama
    if agama is not None:
        return agama
    try:
        import agama as agama_module
    except ImportError:
        sys.path += [os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))]
        try:
            import agama as agama_module
        except ImportError:
            raise ImportError("Could not import the Agama Python extension; build it with "
                "`make` or install it with `pip install . --no-build-isolation`")
    agama = agama_module
    return agama


def as_float(params, name, default=None):
    if name in params:
        return float(params[name])
    if default is not None:
        return default
    raise KeyError("Missing required parameter '%s'" % name)


def gas_surface_density(gas_params, R):
    """Surface density profile matching Agama's built-in Disk convention."""
    sigma0 = as_float(gas_params, "surfaceDensity")
    rdisk = as_float(gas_params, "scaleRadius")
    rcut = as_float(gas_params, "innerCutoffRadius", 0.0)
    sersic_index = as_float(gas_params, "sersicIndex", 1.0)
    rsafe = numpy.maximum(R, 1e-12)
    return sigma0 * numpy.exp(-(rsafe / rdisk)**(1.0 / sersic_index) - rcut / rsafe)


def magnetic_pressure_fraction(plasma_beta):
    """Return P_mag/P_gas for a constant-beta toroidal field."""
    if plasma_beta is None or numpy.isinf(plasma_beta):
        return 0.0
    if not plasma_beta > 0:
        raise ValueError("plasma_beta must be positive, or inf to disable the magnetic field")
    return 1.0 / plasma_beta


def bphi_microgauss(rho_msun_kpc3, sound_speed, plasma_beta):
    """Toroidal magnetic-field amplitude for constant plasma beta."""
    mag_fraction = magnetic_pressure_fraction(plasma_beta)
    if mag_fraction == 0:
        return numpy.zeros_like(rho_msun_kpc3)
    rho_cgs = numpy.maximum(rho_msun_kpc3, 0.0) * MSUN_CGS / KPC_CGS**3
    cs_cgs = sound_speed * KM_CGS
    b_gauss = (8.0 * numpy.pi * rho_cgs * cs_cgs**2 * mag_fraction)**0.5
    return b_gauss * 1.0e6


def magnetic_field_cartesian_microgauss(points, density, sound_speed, plasma_beta):
    """Return Bx, By, Bz for a positive toroidal field in microgauss."""
    points = numpy.atleast_2d(points)
    radius = (points[:, 0]**2 + points[:, 1]**2)**0.5
    bphi = bphi_microgauss(density.density(points), sound_speed, plasma_beta)
    inv_radius = numpy.zeros_like(radius)
    numpy.divide(1.0, radius, out=inv_radius, where=radius > 0)
    return numpy.column_stack((
        -points[:, 1] * inv_radius * bphi,
        points[:, 0] * inv_radius * bphi,
        bphi * 0,
    ))


def make_isothermal_gas_density(
    potential,
    gas_params,
    sound_speed,
    plasma_beta,
    rmin,
    rmax,
    zmin,
    zmax,
    grid_size_r,
    grid_size_z,
    norm_grid_size_r,
    norm_grid_size_z,
    vertical_taper,
):
    """
    Return a DensityAzimuthalHarmonic representation of a vertically isothermal gas disk.

    A globally isothermal atmosphere in a finite-mass potential does not naturally have
    finite vertical extent, so we apply a very broad high-|z| taper.  With the default
    parameters this only affects regions far above the cold gas layer, but keeps the
    density expansion well behaved and finite-mass.
    """
    cs2 = sound_speed * sound_speed
    if not cs2 > 0:
        raise ValueError("sound_speed must be positive")
    ceff2 = cs2 * (1.0 + magnetic_pressure_fraction(plasma_beta))

    grid_r = numpy.geomspace(max(rmin, 1e-4), rmax, norm_grid_size_r)
    grid_z = numpy.linspace(0.0, zmax, norm_grid_size_z)
    rr, zz = numpy.meshgrid(grid_r, grid_z, indexing="ij")
    xyz = numpy.column_stack((rr.ravel(), rr.ravel() * 0, zz.ravel()))
    phi = potential.potential(xyz).reshape(len(grid_r), len(grid_z))
    dphi = numpy.maximum(phi - phi[:, 0:1], 0.0)
    weight = numpy.exp(-numpy.minimum(dphi / ceff2, 700.0))
    if vertical_taper > 0:
        weight *= numpy.exp(-(grid_z / vertical_taper)**4)[None, :]
    znorm = 2.0 * numpy.trapezoid(weight, grid_z, axis=1)
    log_grid_r = numpy.log(grid_r)
    log_znorm = numpy.log(numpy.maximum(znorm, 1e-300))

    def density_function(points):
        points = numpy.atleast_2d(points)
        radius = (points[:, 0]**2 + points[:, 1]**2)**0.5
        rsafe = numpy.maximum(radius, 1e-8)
        rnorm = numpy.clip(rsafe, grid_r[0], grid_r[-1])
        zabs = numpy.abs(points[:, 2])

        xyz_here = numpy.column_stack((rsafe, rsafe * 0, zabs))
        xyz_mid = numpy.column_stack((rsafe, rsafe * 0, rsafe * 0))
        dphi_here = numpy.maximum(potential.potential(xyz_here) - potential.potential(xyz_mid), 0.0)
        hydro_weight = numpy.exp(-numpy.minimum(dphi_here / ceff2, 700.0))
        if vertical_taper > 0:
            hydro_weight *= numpy.exp(-(zabs / vertical_taper)**4)

        sigma = gas_surface_density(gas_params, rsafe)
        zint = numpy.exp(numpy.interp(numpy.log(rnorm), log_grid_r, log_znorm))
        return sigma * hydro_weight / zint

    with agama.setNumThreads(1):
        return agama.Density(
            density=density_function,
            type="DensityAzimuthalHarmonic",
            symmetry="a",
            mmax=0,
            gridsizer=grid_size_r,
            rmin=rmin,
            rmax=rmax,
            gridsizez=grid_size_z,
            zmin=zmin,
            zmax=zmax,
            fixOrder=True,
        )


def gas_rotation_velocity(density, potential, sound_speed, plasma_beta, R, z):
    """
    Return the azimuthal streaming velocity from radial MHD force balance.

    For constant plasma beta, the radial Euler equation gives

        v_phi^2 = R dPhi/dR
                  + R c_s^2 (1 + 1/beta_B) d ln(rho) / dR
                  + 2 c_s^2 / beta_B.

    This is the same expression used for particle velocities, but evaluated on
    arbitrary cylindrical coordinates for grid/profile output.
    """
    R = numpy.asarray(R)
    z = numpy.asarray(z)
    mag_fraction = magnetic_pressure_fraction(plasma_beta)
    cs2 = sound_speed * sound_speed
    ceff2 = cs2 * (1.0 + mag_fraction)
    rsafe = numpy.maximum(R, 1e-6)
    d_r = numpy.maximum(1e-3 * rsafe, 1e-4)
    rplus = rsafe + d_r
    rminus = numpy.maximum(rsafe - d_r, 1e-6)

    xyz_plus = numpy.column_stack((rplus.ravel(), rplus.ravel() * 0, z.ravel()))
    xyz_minus = numpy.column_stack((rminus.ravel(), rminus.ravel() * 0, z.ravel()))
    rho_plus = numpy.maximum(density.density(xyz_plus), 1e-300)
    rho_minus = numpy.maximum(density.density(xyz_minus), 1e-300)
    dlnrho_dr = (numpy.log(rho_plus) - numpy.log(rho_minus)) / (rplus.ravel() - rminus.ravel())

    xyz_cyl = numpy.column_stack((rsafe.ravel(), rsafe.ravel() * 0, z.ravel()))
    dphi_dr = -potential.force(xyz_cyl)[:, 0]
    vphi2 = R.ravel() * (dphi_dr + ceff2 * dlnrho_dr) + 2.0 * cs2 * mag_fraction
    vphi = numpy.sqrt(numpy.maximum(vphi2, 0.0)).reshape(R.shape)
    return numpy.where(R > 0, vphi, 0.0)


def sample_isothermal_gas(density, potential, sound_speed, plasma_beta, num_particles):
    """
    Sample gas positions and assign pressure-supported azimuthal streaming velocities.

    For constant plasma beta, the radial Euler equation gives

        v_phi^2 = R dPhi/dR
                  + R c_s^2 (1 + 1/beta_B) d ln(rho) / dR
                  + 2 c_s^2 / beta_B.

    The output is suitable as a positional/velocity starting point; internal energy,
    smoothing lengths, metallicities, etc. still need to be added by a hydro-IC writer.
    The ordered field is written separately by write_gas_magnetic_field().
    """
    pos, mass = density.sample(num_particles)
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    radius = (x*x + y*y)**0.5
    rsafe = numpy.maximum(radius, 1e-6)
    vphi = gas_rotation_velocity(density, potential, sound_speed, plasma_beta, radius, z)

    cosphi = numpy.divide(x, rsafe)
    sinphi = numpy.divide(y, rsafe)
    vel = numpy.column_stack((-sinphi * vphi, cosphi * vphi, vphi * 0))
    return numpy.column_stack((pos, vel)), mass


def write_gas_magnetic_field(filename, snapshot, density, sound_speed, plasma_beta):
    """Write a companion text file with the ordered toroidal field at gas particle positions."""
    points = snapshot[:, 0:3]
    bcart = magnetic_field_cartesian_microgauss(points, density, sound_speed, plasma_beta)
    bphi = bphi_microgauss(density.density(points), sound_speed, plasma_beta)
    numpy.savetxt(
        filename,
        numpy.column_stack((points, bcart, bphi)),
        fmt="%.8g",
        delimiter="\t",
        header="x[kpc]\ty[kpc]\tz[kpc]\tBx[microG]\tBy[microG]\tBz[microG]\tBphi[microG]",
    )


def write_rotation_curve(filename, potential):
    radii = numpy.logspace(-2.0, 2.0, 81)
    xyz = numpy.column_stack((radii, radii * 0, radii * 0))
    vcirc = (-potential.force(xyz)[:, 0] * radii)**0.5
    numpy.savetxt(
        filename,
        numpy.column_stack((radii, vcirc)),
        fmt="%.6g",
        delimiter="\t",
        header="radius[kpc]\tv_circ[km/s]",
    )


def write_gas_field_grid(filename, density, potential, sound_speed, plasma_beta,
                         rmin, rmax, nr, zmax, nz):
    """Write an axisymmetric meridional grid for AMR interpolation."""
    radii = numpy.linspace(rmin, rmax, nr)
    heights = numpy.linspace(-zmax, zmax, nz)
    rr, zz = numpy.meshgrid(radii, heights, indexing="ij")
    points = numpy.column_stack((rr.ravel(), rr.ravel() * 0, zz.ravel()))
    rho = density.density(points).reshape(rr.shape)
    bphi = bphi_microgauss(rho, sound_speed, plasma_beta)
    vphi = gas_rotation_velocity(density, potential, sound_speed, plasma_beta, rr, zz)
    numpy.savetxt(
        filename,
        numpy.column_stack((rr.ravel(), zz.ravel(), rho.ravel(), vphi.ravel(), bphi.ravel())),
        fmt="%.8g",
        delimiter="\t",
        header="R[kpc]\tz[kpc]\trho_gas[Msun/kpc^3]\tv_phi[km/s]\tBphi[microG]",
    )


def write_gas_model_formula(filename, gas_params, sound_speed, plasma_beta, zmax, vertical_taper):
    mag_fraction = magnetic_pressure_fraction(plasma_beta)
    ceff2 = sound_speed * sound_speed * (1.0 + mag_fraction)
    with open(filename, "w") as file:
        file.write("# Isothermal gas model used by example_self_consistent_model_isothermal_gas.py\n")
        file.write("# Units: R,z in kpc; velocities in km/s; density in Msun/kpc^3.\n")
        file.write("Sigma_g(R) = Sigma0 * exp[-(R/Rd)^(1/n) - Rcut/R]\n")
        file.write("rho_g(R,z) = Sigma_g(R) * exp[-(Phi(R,z)-Phi(R,0))/ceff^2] * taper(z) / Z(R)\n")
        file.write("Z(R) = 2 * integral_0^zmax exp[-(Phi(R,z)-Phi(R,0))/ceff^2] * taper(z) dz\n")
        file.write("taper(z) = exp[-(|z|/ztaper)^4]\n")
        file.write("Bphi(R,z) = sqrt(8*pi*rho_g*c_s^2/beta_B), converted to microgauss\n")
        file.write("vphi^2(R,z) = R*dPhi/dR + R*ceff^2*dlnrho_g/dR + 2*c_s^2/beta_B\n")
        file.write("\n")
        file.write("Sigma0 = %.17g Msun/kpc^2\n" % as_float(gas_params, "surfaceDensity"))
        file.write("Rd = %.17g kpc\n" % as_float(gas_params, "scaleRadius"))
        file.write("Rcut = %.17g kpc\n" % as_float(gas_params, "innerCutoffRadius", 0.0))
        file.write("n = %.17g\n" % as_float(gas_params, "sersicIndex", 1.0))
        file.write("c_s = %.17g km/s\n" % sound_speed)
        file.write("beta_B = %.17g\n" % plasma_beta)
        file.write("Pmag/Pgas = %.17g\n" % mag_fraction)
        file.write("ceff^2 = %.17g (km/s)^2\n" % ceff2)
        file.write("zmax = %.17g kpc\n" % zmax)
        file.write("ztaper = %.17g kpc\n" % vertical_taper)
        file.write("Phi is the final self-consistent Agama potential exported in *_potential.\n")
        file.write("rho_g is also exported as an Agama DensityAzimuthalHarmonic in *_gas_density.\n")


def write_gas_profiles(prefix, density, gas_params, solar_radius, sound_speed, plasma_beta):
    radii = numpy.hstack(([0.25, 0.5], numpy.linspace(1, 30, 59)))
    xy = numpy.column_stack((radii, radii * 0))
    sigma_model = density.projectedDensity(xy)
    sigma_target = gas_surface_density(gas_params, radii)
    numpy.savetxt(
        prefix + "_gas_surface_density.txt",
        numpy.column_stack((radii, sigma_model, sigma_target)),
        fmt="%.6g",
        delimiter="\t",
        header="R[kpc]\tSigma_model[Msun/kpc^2]\tSigma_target[Msun/kpc^2]",
    )

    height = numpy.hstack((numpy.linspace(0, 1.5, 31), numpy.linspace(1.75, 8, 26)))
    xyz = numpy.column_stack((height * 0 + solar_radius, height * 0, height))
    rho = density.density(xyz)
    bphi = bphi_microgauss(rho, sound_speed, plasma_beta)
    numpy.savetxt(
        prefix + "_gas_vertical_density.txt",
        numpy.column_stack((height, rho, bphi)),
        fmt="%.6g",
        delimiter="\t",
        header="z[kpc]\trho_gas(Rsolar,z)[Msun/kpc^3]\tBphi[microG]",
    )


def print_component_info(model, label, solar_radius):
    print("\n%s" % label)
    for name, index in (("stellar disk+halo", 0), ("bulge", 1), ("dark halo", 2), ("gas", 3)):
        dens = model.components[index].density
        rho0 = dens.density((solar_radius, 0, 0)) * 1e-9
        rho1 = dens.density((solar_radius, 0, 1)) * 1e-9
        print("%-18s M=% .6g Msun, rho(Rsun,0)=% .6g, rho(Rsun,1kpc)=% .6g Msun/pc^3" %
              (name, dens.totalMass(), rho0, rho1))
    print("Potential at origin=-(%g km/s)^2, total mass=%g Msun" %
          ((-model.potential.potential(0, 0, 0))**0.5, model.potential.totalMass()))


def build_parser():
    default_ini = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "SCM.ini"))
    parser = argparse.ArgumentParser(
        description="Self-consistent Agama MW-like model with magnetized hydrostatic isothermal gas."
    )
    parser.add_argument("--ini", default=default_ini, help="Input INI file; default: data/SCM.ini")
    parser.add_argument("--iterations", type=int, default=6, help="Number of DF+gas fixed-point iterations")
    parser.add_argument("--gas-sound-speed", type=float, default=10.0, help="Isothermal gas sound speed [km/s]")
    parser.add_argument("--gas-plasma-beta", type=float, default=float("inf"),
        help="Constant plasma beta P_gas/P_mag for an ordered toroidal field; inf disables B")
    parser.add_argument("--gas-zmax", type=float, default=None, help="Vertical range for gas normalization [kpc]")
    parser.add_argument("--gas-vertical-taper", type=float, default=None,
        help="High-|z| taper scale [kpc]; default: gas-zmax")
    parser.add_argument("--gas-norm-grid-r", type=int, default=160,
        help="R grid size for normalizing the hydrostatic gas column")
    parser.add_argument("--gas-norm-grid-z", type=int, default=512,
        help="z grid size for normalizing the hydrostatic gas column")
    parser.add_argument("--gas-grid-r", type=int, default=None,
        help="R grid size for the gas DensityAzimuthalHarmonic expansion")
    parser.add_argument("--gas-grid-z", type=int, default=None,
        help="z grid size for the gas DensityAzimuthalHarmonic expansion")
    parser.add_argument("--gas-grid-zmin", type=float, default=0.005,
        help="Minimum z for the gas density expansion grid [kpc]")
    parser.add_argument("--profile-rmin", type=float, default=0.0, help="Minimum R for AMR profile table [kpc]")
    parser.add_argument("--profile-rmax", type=float, default=30.0, help="Maximum R for AMR profile table [kpc]")
    parser.add_argument("--profile-zmax", type=float, default=5.0, help="Maximum |z| for AMR profile table [kpc]")
    parser.add_argument("--profile-grid-r", type=int, default=301, help="R grid size for AMR profile table")
    parser.add_argument("--profile-grid-z", type=int, default=201, help="z grid size for AMR profile table")
    parser.add_argument("--format", default="text", choices=("text", "nemo", "gadget"),
        help="Snapshot format accepted by agama.writeSnapshot")
    parser.add_argument("--output-prefix", default="isothermal_gas_model", help="Prefix for output files")
    parser.add_argument("--n-halo", type=int, default=800000, help="Number of dark matter particles")
    parser.add_argument("--n-stars", type=int, default=200000, help="Number of stellar particles")
    parser.add_argument("--n-gas", type=int, default=24000, help="Number of gas particles")
    parser.add_argument("--write-particle-snapshots", action="store_true",
        help="Also sample and write DM, stellar, and gas particle snapshots")
    parser.add_argument("--skip-snapshots", action="store_false", dest="write_particle_snapshots",
        help=argparse.SUPPRESS)
    return parser


def main():
    args = build_parser().parse_args()
    import_agama()

    ini = RawConfigParser()
    ini.optionxform = str
    if not ini.read(args.ini):
        raise RuntimeError("Could not read INI file: %s" % args.ini)

    ini_poten_thin_disk = dict(ini.items("Potential thin disk"))
    ini_poten_thick_disk = dict(ini.items("Potential thick disk"))
    ini_poten_gas_disk = dict(ini.items("Potential gas disk"))
    ini_poten_bulge = dict(ini.items("Potential bulge"))
    ini_poten_dark_halo = dict(ini.items("Potential dark halo"))
    ini_df_thin_disk = dict(ini.items("DF thin disk"))
    ini_df_thick_disk = dict(ini.items("DF thick disk"))
    ini_df_stellar_halo = dict(ini.items("DF stellar halo"))
    ini_df_dark_halo = dict(ini.items("DF dark halo"))
    ini_df_bulge = dict(ini.items("DF bulge"))
    ini_scm_halo = dict(ini.items("SelfConsistentModel halo"))
    ini_scm_bulge = dict(ini.items("SelfConsistentModel bulge"))
    ini_scm_disk = dict(ini.items("SelfConsistentModel disk"))
    ini_scm = dict(ini.items("SelfConsistentModel"))
    solar_radius = ini.getfloat("Data", "SolarRadius")

    agama.setUnits(length=1, velocity=1, mass=1)

    model = agama.SelfConsistentModel(**ini_scm)

    density_bulge = agama.Density(**ini_poten_bulge)
    density_dark_halo = agama.Density(**ini_poten_dark_halo)
    density_thin_disk = agama.Density(**ini_poten_thin_disk)
    density_thick_disk = agama.Density(**ini_poten_thick_disk)
    density_gas_disk = agama.Density(**ini_poten_gas_disk)
    density_stellar_disk = agama.Density(density_thin_disk, density_thick_disk)

    model.components.append(agama.Component(density=density_stellar_disk, disklike=True))
    model.components.append(agama.Component(density=density_bulge, disklike=False))
    model.components.append(agama.Component(density=density_dark_halo, disklike=False))
    model.components.append(agama.Component(density=density_gas_disk, disklike=True))

    print("Computing initial potential from analytic density guesses")
    model.iterate()
    print_component_info(model, "Initial analytic-density model", solar_radius)

    df_halo = agama.DistributionFunction(**ini_df_dark_halo)
    df_bulge = agama.DistributionFunction(**ini_df_bulge)
    df_thin_disk = agama.DistributionFunction(potential=model.potential, **ini_df_thin_disk)
    df_thick_disk = agama.DistributionFunction(potential=model.potential, **ini_df_thick_disk)
    df_stellar_halo = agama.DistributionFunction(**ini_df_stellar_halo)
    df_stellar = agama.DistributionFunction(df_thin_disk, df_thick_disk, df_stellar_halo)
    df_stellar_all = agama.DistributionFunction(df_thin_disk, df_thick_disk, df_stellar_halo, df_bulge)

    model.components[0] = agama.Component(df=df_stellar, disklike=True, **ini_scm_disk)
    model.components[1] = agama.Component(df=df_bulge, disklike=False, **ini_scm_bulge)
    model.components[2] = agama.Component(df=df_halo, disklike=False, **ini_scm_halo)

    gas_rmin = max(as_float(ini_scm, "RminCyl", 0.1) * 0.25, 1e-3)
    gas_rmax = as_float(ini_scm, "RmaxCyl", 50.0)
    gas_zmax = args.gas_zmax if args.gas_zmax is not None else as_float(ini_scm, "zmaxCyl", 10.0)
    gas_taper = args.gas_vertical_taper if args.gas_vertical_taper is not None else gas_zmax
    gas_grid_r = args.gas_grid_r if args.gas_grid_r is not None else as_float(ini_scm, "sizeRadialCyl", 30)
    gas_grid_z = args.gas_grid_z if args.gas_grid_z is not None else as_float(ini_scm, "sizeVerticalCyl", 30)

    print("\nMasses of DF components:")
    print("stellar=%g Msun (thin=%g, thick=%g, stellar halo=%g), bulge=%g Msun, dark halo=%g Msun" %
          (df_stellar.totalMass(), df_thin_disk.totalMass(), df_thick_disk.totalMass(),
           df_stellar_halo.totalMass(), df_bulge.totalMass(), df_halo.totalMass()))
    print("Gas sound speed: %g km/s" % args.gas_sound_speed)
    if numpy.isinf(args.gas_plasma_beta):
        print("Gas magnetic field: disabled")
    else:
        print("Gas magnetic field: constant plasma beta=%g, ordered toroidal B_phi" %
              args.gas_plasma_beta)

    for iteration in range(1, args.iterations + 1):
        print("\nStarting isothermal gas iteration #%d" % iteration)
        density_gas_disk = make_isothermal_gas_density(
            potential=model.potential,
            gas_params=ini_poten_gas_disk,
            sound_speed=args.gas_sound_speed,
            plasma_beta=args.gas_plasma_beta,
            rmin=gas_rmin,
            rmax=gas_rmax,
            zmin=args.gas_grid_zmin,
            zmax=gas_zmax,
            grid_size_r=int(gas_grid_r),
            grid_size_z=int(gas_grid_z),
            norm_grid_size_r=args.gas_norm_grid_r,
            norm_grid_size_z=args.gas_norm_grid_z,
            vertical_taper=gas_taper,
        )
        model.components[3] = agama.Component(density=density_gas_disk, disklike=True)
        model.iterate()
        print_component_info(model, "After iteration %d" % iteration, solar_radius)

    prefix = args.output_prefix
    print("\nWriting diagnostics")
    write_rotation_curve(prefix + "_rotation_curve.txt", model.potential)
    write_gas_profiles(prefix, model.components[3].density, ini_poten_gas_disk,
        solar_radius, args.gas_sound_speed, args.gas_plasma_beta)
    write_gas_field_grid(prefix + "_gas_Rz_profile.txt", model.components[3].density,
        model.potential, args.gas_sound_speed, args.gas_plasma_beta,
        args.profile_rmin, args.profile_rmax, args.profile_grid_r,
        args.profile_zmax, args.profile_grid_z)
    write_gas_model_formula(prefix + "_gas_formula.txt", ini_poten_gas_disk,
        args.gas_sound_speed, args.gas_plasma_beta, gas_zmax, gas_taper)
    model.potential.export(prefix + "_potential")
    model.components[3].density.export(prefix + "_gas_density")

    if not args.write_particle_snapshots:
        print("Skipping particle snapshots")
        print("Done")
        return

    print("\nWriting initial-condition snapshots")
    agama.writeSnapshot(
        prefix + "_dm_final",
        agama.GalaxyModel(potential=model.potential, df=df_halo, af=model.af).sample(args.n_halo),
        args.format,
    )
    agama.writeSnapshot(
        prefix + "_stars_final",
        agama.GalaxyModel(potential=model.potential, df=df_stellar_all, af=model.af).sample(args.n_stars),
        args.format,
    )
    gas_snapshot = sample_isothermal_gas(model.components[3].density, model.potential,
        args.gas_sound_speed, args.gas_plasma_beta, args.n_gas)
    agama.writeSnapshot(prefix + "_gas_final", gas_snapshot, args.format)
    write_gas_magnetic_field(prefix + "_gas_bfield.txt", gas_snapshot[0],
        model.components[3].density, args.gas_sound_speed, args.gas_plasma_beta)
    print("Done")


if __name__ == "__main__":
    main()

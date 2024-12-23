"""Constants"""
# Atomic units of Ry

# Conversion factor
Ha2eV = 27.21138386
eV2Ry = 2./Ha2eV
Ry2eV = 1./eV2Ry
a2bohr = 1.88973

# Boltzman constant
kb_HaK = 3.1668154267112283e-06

# Electron mass over atomical mass unit
me_amu = 5.4857990965007152E-4

# We use Ry in energy so time unit = hbar/Ry
# convert time unit to nanosec
# hbar/hartree = 2.418884326505*10^{-17} sec
t2nsec = 2.418884326505*2*1E-8

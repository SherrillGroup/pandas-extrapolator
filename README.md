# pandas-extrapolator
Micro project to do CBS extrapolation on Pandas dataframes

Using the 2-point Helgaker extrapolation, e.g.,
A. Halkier, W. Klopper, T. Helgaker, P. Jorgensen, and P. R. Taylor,
J. Chem. Phys. 111, 9157 (1999)

For 2 cardinal numbers of basis sets (X-1) and (X) (e.g., 3 and 4
for cc-pVTZ and cc-pVQZ), the CBS limit of the ***correlation energy***
can be estimated as

E_{corr, CBS} = [ X^3 E_{corr, X} - (X-1)^3 E_{corr, X-1}] / [ X^3 - (X-1)^3 ]



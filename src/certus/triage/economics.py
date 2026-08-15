"""Human-review fraction (HRF = (|FLAG| + |ESCALATE|) / N) and the cost
break-even model: net saving per image = (1 - HRF(epsilon)) * c_h -
epsilon * pi * c_m, swept over the c_m/c_h ratio to find the region where
automation is economically rational. Feeds Figures F4 and F8. See roadmap
Section 3.4, Section 5.4.
"""

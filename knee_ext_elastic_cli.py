"""
% KNEE_EXT_ELASTIC_CLI simulates a leg extension at an inclined legpress
%
%  It includes a parallel-elastic element,
%  a non-linear to linear serial elastic element, and
%  a force-length relation.
%  The Newtonian equation of motion is expressed as an ODE wich considers
%  initial conditions and neuromuscular properties provided by th user.
%
%  Translated from Matlab R2016b to Python 3.12.0
%  Depencencies:
%    - numpy
%    - scipy
%    - matplotlib
%    - pathlib
%    - datetime 
%    - typing
%    - math
%
%  USAGE
%    Just run the file, maybe define an output folder.
%    Change parameters in the secton "Input" if you like.
%
%  REFERENCE
%   H. Penasso & S. Thaller (2017) Determination of individual knee-extensor
    properties from leg extensions and parameter identification, Mathematical and Computer
    Modelling of Dynamical Systems, 23:4, 416-438, DOI: 10.1080/13873954.2017.1336633
%
%  INPUT
%    has to be defined below in section "Input"
%
%  OUTPUT
%    A txt-file is saved and contains the simulated data as well as
%    the settings, the inital conditions and the properties of the system.
%    t  ...   Time [s]
%    F  ...   External force [N]
%    X  ...   Position: distance from the prox. end of the model-thigh to
%             the proximal end of the model-shank [m]
%    V  ...   Velocitiy of the accelerated point-mass [m/s]
%   dV  ...   Acceleration of the point-mass [m/s^2]
% v_SEE ...   Velocity of the tendon-model [m/s]
%  f_CE ...   Values of the contraction-velocity dependent element (CE) [N]
%  v_CE ...   Contraction velocity of the musle (CE=MUSCLE=PEE) [m/s]
% f_PEE ...   Force of passive muscle tissue PEE [N]
% f_MTC ...   Force of the muscle-tendon complex (CE=MUSCLE=SEE) [N]
% v_MTC ...   Velocity of the muscle-tendon complex [m/s]
%   GX  ...   Values of the function of geometry [-]
%   AT  ...   Values of the function of activation dynamics [-]
%   FL  ...   Values of the force-rength relation [-]
% l_SEE ...   Length of the serial elastic element [m]
%  l_CE ...   Length of the contractile and parallel elsatic element [m]
% l_MTC ...   Length of the mucle-tendon complex [m]
%
%  Literature:
%   [1] Hoy, M. G., Zajac, F. E., & Gordon, M. E. (1990). A
%      musculoskeletal model of the human lower extremity: The effect
%      of muscle, tendon, and moment arm on the moment-angle
%      relationship of musculotendon actuators at the hip, knee, and
%      ankle. Journal of Biomechanics, 23(2), 157?169.
%      doi:10.1016/0021-9290(90)90349-8
%   [2] Im, H., Goltzer, O., & Sheehan, F. (2015). The effective quadriceps
%      and patellar tendon moment arms Relative to the Tibiofemoral Finite
%      Helical Axis. Journal of Biomechanics.
%      doi:10.1016/j.jbiomech.2015.04.003
%   [3] Van Eijden, T., Kouwenhoven, E., Verburg, J., & Weijs, W. (1986).
%      A mathematical model of the patellofemoral joint. Journal of
%      Biomechanics, 19(3), 219?229. doi:10.1016/0021-9290(86)90154-5
%   [4] Van Soest, A. J., & Bobbert, M. F. (1993). The contribution of
%      muscle properties in the control of explosive movements.
%      Biological Cybernetics, 69(3), 195?204.
%      Retrieved from http://www.ncbi.nlm.nih.gov/pubmed/8373890
%   [5] http://www.mathworks.com/matlabcentral/fileexchange/
%      11077-figure-digitizer/content/figdi.m
%   [6] G?nther, M., Schmitt, S., & Wank, V. (2007). High-frequency
%      oscillations as a consequence of neglected serial damping in
%      Hill-type muscle models. Biological Cybernetics, 97(1), 63?79.
%      doi:10.1007/s00422-007-0160-6
%   [7] Haeufle, D. F. B., Guenther, M., Bayer, A., & Schmitt, S. (2014).
%      Hill-type muscle model with serial damping and eccentric
%      force-velocity relation. Journal of Biomechanics, 47(6), 1531?1536.
%      doi:10.1016/j.jbiomech.2014.02.009
%   [8] Pandy, M. G., Zajac, F. E., Sim, E., & Levine, W. S. (1990).
%      An optimal control model for maximum-height human jumping.
%      Journal of Biomechanics, 23(12), 1185?98.
%      Retrieved from http://www.ncbi.nlm.nih.gov/pubmed/2292598
%  
%  v.5.2 by Harald Penasso (16.11.2015) (cleaned up: 24.11.2016)
%                          (translated to Python using ChatGPT 5.1 w/ Extended Thinking)
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Sequence, Optional, Dict, Any, Tuple
import math

def knee_ext_elastic(
    isoQ: bool = False,
    output_dir: str | Path = ".",
    save_plots: bool = True,
    save_txt: bool = True,
):
    """
    Python translation of the MATLAB KneeExtElastic main function.

    Returns a dict of numpy arrays with keys:
        t, F, X, V, dV, v_SEE, f_CE, v_CE, f_PEE, f_MTC,
        v_MTC, GX, AT, FL, l_SEE, l_CE, l_MTC
    """
    
    # ------------------------------------------------------------------
    # A.0 Helpers-0 (allows flexible parameter conversion)
    # ------------------------------------------------------------------
    
    def _abc_from_vmax_fiso_pmax(vmax: float, fiso: float, pmax: float) -> Tuple[float, float, float]:
        """
        Helper implementing the shared algebra for cases 1, 3, 4:
        given vmax, fiso, pmax, compute the Hill parameters a, b, c.
        """
        vmax = float(vmax)
        fiso = float(fiso)
        pmax = float(pmax)
    
        # In your MATLAB code: sqrt(fiso^-1 * pmax * vmax^-1)
        root = math.sqrt(pmax / (fiso * vmax))
        denom_factor = (1.0 - 2.0 * root)
    
        # a = pmax / (vmax * (1 - 2*root))
        a = pmax / (vmax * denom_factor)
    
        # b = pmax / (fiso * (1 - 2*root))
        b = pmax / (fiso * denom_factor)
    
        # c = (1/fiso)*pmax*(1-2*root)^(-2) * (1/vmax) * (pmax + fiso*(1-2*root)*vmax)
        c = (pmax / fiso) * (1.0 - 2.0 * root) ** -2 * (1.0 / vmax) * (
            pmax + fiso * (1.0 - 2.0 * root) * vmax
        )
    
        return a, b, c
    
    def params_calc(
        hill_model: int,
        par: Sequence[float],
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float, float]:
        """
        Exact 1:1 translation of the MATLAB:
    
            [U,a,b,c,vmax,fiso,pmax,vopt,fopt,eta,kappa] = params(HillModell, Par)
    
        No algebraic simplifications.
        """
    
        hm = int(hill_model)
    
        if hm == 1:
            # case 1 % pmax, vmax, fiso
            U = float(par[0])
            vmax = float(par[1])
            fiso = float(par[2])
            pmax = float(par[3])
    
            a = pmax * (
                vmax
                + (-2.0)
                * (fiso ** (-1.0) * pmax * vmax ** (-1.0)) ** 0.5
                * vmax
            ) ** (-1.0)
    
            b = pmax * (
                fiso
                + (-2.0)
                * fiso
                * (fiso ** (-1.0) * pmax * vmax ** (-1.0)) ** 0.5
            ) ** (-1.0)
    
            c = (
                fiso ** (-1.0)
                * pmax
                * (
                    1.0
                    + (-2.0)
                    * (fiso ** (-1.0) * pmax * vmax ** (-1.0)) ** 0.5
                ) ** (-2.0)
                * vmax ** (-1.0)
                * (
                    pmax
                    + fiso
                    * (
                        1.0
                        + (-2.0)
                        * (fiso ** (-1.0) * pmax * vmax ** (-1.0)) ** 0.5
                    )
                    * vmax
                )
            )
    
            vopt = math.sqrt((b * c) / a) - b
            fopt = math.sqrt((a * c) / b) - a
            eta = pmax / c
            kappa = a / fiso
    
        elif hm == 2:
            # case 2 % a, b, c
            U = float(par[0])
            a = float(par[1])
            b = float(par[2])
            c = float(par[3])
    
            vmax = c / a - b
            fiso = c / b - a
            pmax = a * b + c - 2.0 * math.sqrt(a * b * c)
            vopt = math.sqrt((b * c) / a) - b
            fopt = math.sqrt((a * c) / b) - a
            eta = pmax / c
            kappa = a / fiso
    
        elif hm == 3:
            # case 3 % vmax, fiso, eta
            U = float(par[0])
            vmax = float(par[1])
            fiso = float(par[2])
            eta = float(par[3])
    
            pmax = (eta - 4.0) ** (-2.0) * (
                (4.0 + (-3.0 + eta) * eta) * fiso * vmax
                + (-2.0)
                * ((eta - 2.0) ** 2.0 * eta * fiso ** 2.0 * vmax ** 2.0) ** 0.5
            )
    
            a = pmax * (
                vmax
                + (-2.0)
                * (fiso ** (-1.0) * pmax * vmax ** (-1.0)) ** 0.5
                * vmax
            ) ** (-1.0)
    
            b = pmax * (
                fiso
                + (-2.0)
                * fiso
                * (fiso ** (-1.0) * pmax * vmax ** (-1.0)) ** 0.5
            ) ** (-1.0)
    
            c = (
                fiso ** (-1.0)
                * pmax
                * (
                    1.0
                    + (-2.0)
                    * (fiso ** (-1.0) * pmax * vmax ** (-1.0)) ** 0.5
                ) ** (-2.0)
                * vmax ** (-1.0)
                * (
                    pmax
                    + fiso
                    * (
                        1.0
                        + (-2.0)
                        * (fiso ** (-1.0) * pmax * vmax ** (-1.0)) ** 0.5
                    )
                    * vmax
                )
            )
    
            vopt = math.sqrt((b * c) / a) - b
            fopt = math.sqrt((a * c) / b) - a
            # eta was input, keep it
            kappa = a / fiso
    
        elif hm == 4:
            # case 4 % vmax, eta, pmax
            U = float(par[0])
            vmax = float(par[1])
            eta = float(par[2])
            pmax = float(par[3])
    
            fiso = (eta - 1.0) ** (-2.0) * vmax ** (-2.0) * (
                (4.0 + (-3.0 + eta) * eta) * pmax * vmax
                + 2.0
                * ((eta - 2.0) ** 2.0 * eta * pmax ** 2.0 * vmax ** 2.0) ** 0.5
            )
    
            a = pmax * (
                vmax
                + (-2.0)
                * (fiso ** (-1.0) * pmax * vmax ** (-1.0)) ** 0.5
                * vmax
            ) ** (-1.0)
    
            b = pmax * (
                fiso
                + (-2.0)
                * fiso
                * (fiso ** (-1.0) * pmax * vmax ** (-1.0)) ** 0.5
            ) ** (-1.0)
    
            c = (
                fiso ** (-1.0)
                * pmax
                * (
                    1.0
                    + (-2.0)
                    * (fiso ** (-1.0) * pmax * vmax ** (-1.0)) ** 0.5
                ) ** (-2.0)
                * vmax ** (-1.0)
                * (
                    pmax
                    + fiso
                    * (
                        1.0
                        + (-2.0)
                        * (fiso ** (-1.0) * pmax * vmax ** (-1.0)) ** 0.5
                    )
                    * vmax
                )
            )
    
            vopt = math.sqrt((b * c) / a) - b
            fopt = math.sqrt((a * c) / b) - a
            # eta was input, keep it
            kappa = a / fiso
    
        elif hm == 5:
            # case 5 % vopt, fopt, fiso
            U = float(par[0])
            vopt = float(par[1])
            fopt = float(par[2])
            fiso = float(par[3])
    
            a = (fiso + (-2.0) * fopt) ** (-1.0) * fopt ** 2.0
            b = (fiso + (-2.0) * fopt) ** (-1.0) * fopt * vopt
            c = (fiso + (-2.0) * fopt) ** (-2.0) * (fiso + (-1.0) * fopt) ** 2.0 * fopt * vopt
    
            vmax = c / a - b
            pmax = (fopt ** 2.0) * b / a
            kappa = a / fiso
            eta = pmax / c
    
        else:
            raise ValueError(f"Unknown hill_model {hill_model}; expected 1–5.")
    
        return U, a, b, c, vmax, fiso, pmax, vopt, fopt, eta, kappa
    
    def params(
        *,
        U: float = 1.0,
        hill_model: Optional[int] = None,
        vmax: Optional[float] = None,
        fiso: Optional[float] = None,
        pmax: Optional[float] = None,
        eta: Optional[float] = None,
        vopt: Optional[float] = None,
        fopt: Optional[float] = None,
        a: Optional[float] = None,
        b: Optional[float] = None,
        c: Optional[float] = None,
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float, float]:
        """
        Convenience wrapper that chooses the Hill model representation
        based on which keyword arguments are provided.
    
        Valid combinations (besides U):
    
          hill_model=1 -> vmax, fiso, pmax
          hill_model=2 -> a, b, c
          hill_model=3 -> vmax, fiso, eta
          hill_model=4 -> vmax, eta, pmax
          hill_model=5 -> vopt, fopt, fiso
    
        If `hill_model` is None, the function tries to infer it from
        these combinations. If more than one combination matches, it
        raises an error.
        """
        combos = {
            1: ("vmax", "fiso", "pmax"),
            2: ("a", "b", "c"),
            3: ("vmax", "fiso", "eta"),
            4: ("vmax", "eta", "pmax"),
            5: ("vopt", "fopt", "fiso"),
        }
    
        values: Dict[str, Any] = dict(
            vmax=vmax, fiso=fiso, pmax=pmax, eta=eta,
            vopt=vopt, fopt=fopt, a=a, b=b, c=c
        )
    
        def has_all(names):
            return all(values[name] is not None for name in names)
    
        # infer hill_model if not given
        if hill_model is None:
            valid = [m for m, req in combos.items() if has_all(req)]
            if not valid:
                raise ValueError(
                    "Cannot infer hill_model from provided arguments. "
                    "Need one of the combinations: "
                    "(vmax, fiso, pmax), (a, b, c), (vmax, fiso, eta), "
                    "(vmax, eta, pmax), or (vopt, fopt, fiso)."
                )
            if len(valid) > 1:
                raise ValueError(
                    f"Ambiguous parameter combination; matches hill models {valid}. "
                    "Specify hill_model explicitly."
                )
            hill_model = valid[0]
    
        hm = int(hill_model)
    
        # Build Par vector in the same convention as params_calc()
        if hm == 1:
            if not has_all(combos[1]):
                raise ValueError("hill_model=1 requires vmax, fiso, pmax.")
            par = [U, vmax, fiso, pmax]
    
        elif hm == 2:
            if not has_all(combos[2]):
                raise ValueError("hill_model=2 requires a, b, c.")
            par = [U, a, b, c]
    
        elif hm == 3:
            if not has_all(combos[3]):
                raise ValueError("hill_model=3 requires vmax, fiso, eta.")
            par = [U, vmax, fiso, eta]
    
        elif hm == 4:
            if not has_all(combos[4]):
                raise ValueError("hill_model=4 requires vmax, eta, pmax.")
            par = [U, vmax, eta, pmax]
    
        elif hm == 5:
            if not has_all(combos[5]):
                raise ValueError("hill_model=5 requires vopt, fopt, fiso.")
            par = [U, vopt, fopt, fiso]
    
        else:
            raise ValueError(f"Unknown hill_model {hill_model}; expected 1–5.")
    
        return params_calc(hm, par)
        
    # ------------------------------------------------------------------
    # A.1 Input (options and parameters, following the MATLAB code)
    # ------------------------------------------------------------------
    t0 = 0.0
    te = 6.5
    reltol = 1e-8
    maxstep = 0.005

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initial conditions and neuromuscular properties
    isoQ = bool(isoQ)  # isometric condition flag
    phi = 90.0  # plane inclination [deg]
    g_earth = 9.81  # [m/s^2]
    msledge = 32.5  # [kg]
    add = 60.0  # additional mass [kg]
    KA0 = 120.0  # initial knee-extension angle [deg]
    V0 = 0.0  # initial velocity [m/s]
    dV0 = 0.0  # initial acceleration [m/s^2]
    t_on = 0.25  # innervation start [s]

    # Geometry
    lt = 0.43  # thigh length [m]
    ls = 0.43  # shank length [m]
    cir = 0.37  # knee circumference [m]
    ptl = 0.08  # patellar tendon length [m]

    # Muscle-tendon complex parameters
    # Activation dynamics
    U = 12.0

    # Serial elastic element
    ksee = 1_777_000.0  # [N/m]
    ns = 1.0
    SEE_TH = 1.0  # transition point (fraction of fiso)

    # Parallel elastic element
    kpee = 0.0
    np_pee = 2.0

    # Contractile element
    a = 7.920643997366427e3  # [N]
    b = 0.352028622105175  # [m/s]
    c = 9.124808590471668e3  # [W]
    vmax = 0.8 # [m/s]
    fiso = 18000 # [N]
    pmax = 1824.96 # [W]
    vopt = 0.284797 # [m/s]
    fopt = 6407.94 # [N]
    eta = 0.2 # [-]
    
    # Flexible parameter-set convertion
    """
    U, a, b, c, vmax, fiso, pmax, vopt, fopt, eta, kappa = params(
        a=a,
        b=b,
        c=c,
        U=U,
    )
    """
    U, a, b, c, vmax, fiso, pmax, vopt, fopt, eta, kappa = params(
        vmax=vmax,
        pmax=pmax,
        fiso=fiso,
        U=U,
    )
    
    # Force-length relationship
    KA_opt = 120.0
    WIDTH = 0.56
    l_CE_opt = 0.09

    # ------------------------------------------------------------------
    # Helper functions (sub-functions translated from MATLAB)
    # ------------------------------------------------------------------

    def ES_fun(t, t_on_local, Apre_local, U_local):
        """
        Activation dynamics ES_fun translated from MATLAB.
        Works for scalar or numpy array t.
        """
        t_arr = np.asarray(t, dtype=float)
        A = np.zeros_like(t_arr)
        nmax = 1.0
        umax = U_local

        before = t_arr < t_on_local
        after = ~before

        A[before] = Apre_local

        if np.any(after):
            ta = t_arr[after]
            expr = np.exp(
                -np.exp(-(ta - t_on_local) * U_local)
                * U_local ** (-1.0)
                * (
                    (-Apre_local + umax)
                    + np.exp((ta - t_on_local) * U_local)
                    * (Apre_local + (-1.0 + (ta - t_on_local) * U_local) * umax)
                )
            )
            A[after] = expr * (Apre_local - nmax) + nmax

        if np.isscalar(t):
            return float(A)
        return A

    # Geometry-related helpers -----------------------------------------

    lr = cir / (2.0 * np.pi)  # approximated moment arm [m]
    r_moment = lr
    ko = lt

    def _beta_for_sigma(sigma_val):
        """Find beta that minimizes bfunc, translated from fminbnd loop."""
        def objective(v):
            # clamp argument of arcsin
            arg1 = r_moment * np.sin(v) / ko
            arg2 = r_moment * np.sin(v) / ptl
            arg1 = np.clip(arg1, -1.0, 1.0)
            arg2 = np.clip(arg2, -1.0, 1.0)
            return (
                2 * v
                + np.arcsin(arg1)
                + np.arcsin(arg2)
                - sigma_val
            ) ** 2

        res = minimize_scalar(objective, bounds=(0.0, np.pi), method="bounded")
        return float(res.x)

    KA_opt_rad = np.deg2rad(KA_opt)
    beta_opt = _beta_for_sigma(KA_opt_rad)
    # MTC length at optimal knee angle
    arg_opt = r_moment * np.sin(beta_opt) / lt
    arg_opt = np.clip(arg_opt, -1.0, 1.0)
    l_MTC_opt = np.sqrt(
        lt ** 2
        + r_moment ** 2
        - 2 * lt * r_moment * np.cos(np.pi - beta_opt - np.arcsin(arg_opt))
    )

    def _G_and_dlMTC(X_arr):
        X_arr = np.asarray(X_arr, dtype=float)
        # cos law for knee angle
        cos_arg = (lt ** 2 + ls ** 2 - X_arr ** 2) / (2 * lt * ls)
        cos_arg = np.clip(cos_arg, -1.0, 1.0)
        sigma = np.arccos(cos_arg)

        # beta for each sigma
        beta_vals = np.array([_beta_for_sigma(s) for s in np.atleast_1d(sigma)])

        # MTC length
        arg = r_moment * np.sin(beta_vals) / lt
        arg = np.clip(arg, -1.0, 1.0)
        l_MTC = np.sqrt(
            lt ** 2
            + r_moment ** 2
            - 2 * lt * r_moment * np.cos(np.pi - beta_vals - np.arcsin(arg))
        )

        dl_MTC = l_MTC_opt - l_MTC
        G = r_moment * X_arr * np.sin(beta_vals) / (lt * ls * np.sin(sigma))

        # For X beyond geometric limit, set G = 0
        G = np.where(X_arr > (lt + ls), 0.0, G)
        return G, dl_MTC

    def geometry_fun(X_val, dX_val):
        """
        geometry_fun: returns (G, dG, dl_MTC)

        dG is approximated numerically via a small finite-difference in X,
        using dX (velocity) as direction.
        """
        X_arr = np.atleast_1d(np.array(X_val, dtype=float))
        dX_arr = np.atleast_1d(np.array(dX_val, dtype=float))

        G, dl_MTC = _G_and_dlMTC(X_arr)

        # numerical time derivative of G via finite differences
        eps = 1e-6
        G_plus, _ = _G_and_dlMTC(X_arr + dX_arr * eps)
        G_minus, _ = _G_and_dlMTC(X_arr - dX_arr * eps)
        dG = (G_plus - G_minus) / (2 * eps)

        if np.isscalar(X_val):
            return float(G[0]), float(dG[0]), float(dl_MTC[0])
        return G, dG, dl_MTC

    # Force-length + PEE -----------------------------------------------

    def fl_fpee_lsee_fun(dx, dlmtc, QPTL, l_CE_opt_val, width, k_pee, n_pee):
        dx = np.asarray(dx, dtype=float)
        dlmtc = np.asarray(dlmtc, dtype=float)

        # SEE length
        l_SEE = QPTL + dx
        # CE length
        l_CE = l_CE_opt_val - dx - dlmtc

        l_CE_min = np.min(l_CE)
        if not ((1 - width) * l_CE_opt_val <= l_CE_min <= (1 + width) * l_CE_opt_val):
            raise ValueError("The CE is out of bounds!")

        c_fl = -1.0 / (width ** 2)
        fl = (
            c_fl * (l_CE / l_CE_opt_val) ** 2
            - 2 * c_fl * (l_CE / l_CE_opt_val)
            + c_fl
            + 1.0
        )

        # PEE force
        f_PEE = k_pee * (l_CE - l_CE_opt_val) ** n_pee
        f_PEE = np.where(l_CE < l_CE_opt_val, 0.0, f_PEE)

        return fl, f_PEE, l_SEE, l_CE

    # ------------------------------------------------------------------
    # A.2 Initial calculations
    # ------------------------------------------------------------------
    _, a, b, c, vmax, fiso, pmax, vopt, fopt, eta, kappa = params_calc(2, [U, a, b, c])

    f_MTC_th = fiso * SEE_TH
    dx_th = (ns * f_MTC_th) / ksee
    ksnl = ksee * ns ** (-1.0) * dx_th ** (1.0 - ns)

    # function of geometry initial values
    X0 = np.sqrt(lt ** 2 + ls ** 2 - 2 * lt * ptl * np.cos(np.deg2rad(KA0)))
    g = g_earth * np.sin(np.deg2rad(phi))
    m = msledge + add

    G_val0, _, dlmtc0 = geometry_fun(X0, V0)

    f_MTC0 = m * (dV0 + g) / G_val0

    # initial SEE elongation
    if f_MTC0 > f_MTC_th:
        dx0 = (f_MTC0 + ksee * dx_th * (1.0 - 1.0 / ns)) / ksee
    else:
        dx0 = (f_MTC0 / ksnl) ** (1.0 / ns)

    fL0, f_PEE0, _, _ = fl_fpee_lsee_fun(
        dx0, dlmtc0, ptl, l_CE_opt, WIDTH, kpee, np_pee
    )

    Apre = (b * f_PEE0 * G_val0 - b * g * m) / (
        (a * b * fL0 * G_val0 - c * fL0 * G_val0)
    )

    # ------------------------------------------------------------------
    # A.3 Filename id and checks
    # ------------------------------------------------------------------
    add_for_name = int(add if add < 99 else 99)
    sim_id = f"psim00g{add_for_name:02d}v01"

    # basic validity checks (translated from MATLAB)
    if KA0 < 60.0:
        raise ValueError("The initial knee-extension-angle is too small (KA0 < 60).")

    if a / fiso > 1.0:
        raise ValueError("The curvature a/fiso is > 1.")

    if pmax / c > 0.7:
        raise ValueError("The efficiency is too big (eta > 0.7).")

    if pmax / c < 0.05:
        raise ValueError("The efficiency is too small (eta < 0.05).")

    if te <= t_on:
        raise ValueError("The contraction must begin within [t0, te) (te > t_on).")

    # ------------------------------------------------------------------
    # A.5 ODE solving via solve_ivp
    # ------------------------------------------------------------------

    def model_fun(t, Y):
        X_pos, V_vel, A_acc = Y

        # sanity checks before contraction starts
        if t <= t_on:
            GX0, _, _ = geometry_fun(X_pos, V_vel)
            if GX0 * V0 > vmax:
                raise RuntimeError("Initial contraction-velocity > vmax")

            GX_init, _, _ = geometry_fun(X0, V0)
            if m * g > GX_init * (c / (GX_init * V0 + b) - a):
                raise RuntimeError("Subject cannot hold the weight at initial position")

        GX, dG, dlmtc = geometry_fun(X_pos, V_vel)
        f_MTC = abs(m * (A_acc + g) / GX)

        # SEE elongation
        if f_MTC < f_MTC_th:
            dx = (f_MTC / ksnl) ** (1.0 / ns)
        else:
            dx = (f_MTC + ksee * dx_th * (1.0 - 1.0 / ns)) / ksee

        FL_val, f_PEE_val, _, _ = fl_fpee_lsee_fun(
            dx, dlmtc, ptl, l_CE_opt, WIDTH, kpee, np_pee
        )

        AT_val = ES_fun(t, t_on, Apre, U)

        if isoQ:
            dXdt = 0.0
            dVdt = 0.0
        else:
            dXdt = V_vel
            dVdt = A_acc

        if f_MTC < f_MTC_th:
            denom = (a * AT_val * FL_val * GX - f_PEE_val * GX + m * (g + A_acc))
            term1 = (1.0 / GX) * dG * (g + A_acc)
            coef = -dx_th ** (1.0 - ns) * ksee * GX / m * dx ** (ns - 1.0)
            term2_inner = (
                AT_val
                * FL_val
                * GX
                * (a * b - c + a * GX * V_vel)
                + (b + GX * V_vel)
                * (-f_PEE_val * GX + m * (g + A_acc))
            )
            dAdt = term1 + coef * term2_inner / denom
        else:
            denom = (a * AT_val * FL_val * GX - f_PEE_val * GX + m * (g + A_acc))
            inner = -b + GX * (-V_vel + c * AT_val * FL_val / denom)
            dAdt = (1.0 / GX) * (dG * (g + A_acc) + ksee * GX ** 2 / m * inner)

        return [dXdt, dVdt, dAdt]

    def event_force_zero(t, Y):
        # m * x(3) + m * g
        return m * Y[2] + m * g

    event_force_zero.terminal = True
    event_force_zero.direction = 0

    def event_knee_angle_limit(t, Y):
        X_pos = Y[0]
        # knee angle from cosine law
        cos_arg = (lt ** 2 + ls ** 2 - X_pos ** 2) / (2 * lt * ls)
        cos_arg = np.clip(cos_arg, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_arg))
        return angle_deg - 179.0

    event_knee_angle_limit.terminal = True
    event_knee_angle_limit.direction = 0

    sol = solve_ivp(
        model_fun,
        (t0, te),
        y0=[X0, V0, dV0],
        rtol=reltol,
        max_step=maxstep,
        events=[event_force_zero, event_knee_angle_limit],
    )

    t = sol.t
    X = sol.y[0, :]
    V = sol.y[1, :]
    dV = sol.y[2, :]

    # ------------------------------------------------------------------
    # A.7 Calculate subsystem values
    # ------------------------------------------------------------------
    F = m * dV + m * g

    GX, dG, dlmtc = geometry_fun(X, V)

    f_MTC = m * (dV + g) / GX
    v_MTC = GX * V

    # SEE elongations over whole trajectory
    f_MTC_arr = f_MTC
    dx = np.empty_like(f_MTC_arr)
    mask_lin = f_MTC_arr >= f_MTC_th
    mask_nl = ~mask_lin
    dx[mask_lin] = (f_MTC_arr[mask_lin] + ksee * dx_th * (1.0 - 1.0 / ns)) / ksee
    dx[mask_nl] = np.power(np.abs(f_MTC_arr[mask_nl] / ksnl), 1.0 / ns)

    FL, f_PEE, l_SEE, l_CE = fl_fpee_lsee_fun(
        dx, dlmtc, ptl, l_CE_opt, WIDTH, kpee, np_pee
    )

    f_CE = f_MTC - f_PEE
    AT = ES_fun(t, t_on, Apre, U)

    v_CE = c / ((f_MTC - f_PEE) / (AT * FL) + a) - b
    v_SEE = v_MTC - v_CE
    l_MTC = l_CE + l_SEE

    # ------------------------------------------------------------------
    # A.8 Plotting
    # ------------------------------------------------------------------
    if save_plots:
        fig = plt.figure(figsize=(16, 12))

        # Internal forces
        ax1 = fig.add_subplot(4, 3, (1, 4))
        ax1.plot(t, f_CE, "r.-", label="F_CE")
        ax1.plot(t, f_PEE, "m.-", label="F_PEE")
        ax1.plot(t, f_MTC, "g.-", label="F_MTC")
        ax1.set_xlabel("time [s]")
        ax1.set_ylabel("force [N]")
        ax1.set_title("Internal forces")
        ax1.legend()
        ax1.grid(True)

        # Geometry
        ax2 = fig.add_subplot(4, 3, (2, 5))
        ax2.plot(t, GX, "c.-")
        ax2.set_title("Function of geometry")
        ax2.set_xlabel("time [s]")
        ax2.set_ylabel("ratio [-]")
        ax2.grid(True)

        # Activation dynamics
        ax3 = fig.add_subplot(4, 3, 3)
        ax3.plot(t, AT, "c.-")
        ax3.set_xlabel("time [s]")
        ax3.set_ylabel("ratio [-]")
        ax3.set_ylim(0, 1.1)
        ax3.set_title("Activation dynamics")
        ax3.grid(True)

        # Force-length dynamics
        ax4 = fig.add_subplot(4, 3, 6)
        ax4.plot(t, FL, "k.-", label="fl")
        ax4.set_xlabel("time [s]")
        ax4.set_ylabel("ratio [-]")
        ax4.set_ylim(np.min(FL) * 0.99, 1.01)
        ax4.set_title("Force-length dynamics")
        ax4.legend()
        ax4.grid(True)

        # External force
        ax5 = fig.add_subplot(4, 3, (7, 10))
        ax5.plot(t, F, "b.-")
        ax5.set_xlabel("time [s]")
        ax5.set_ylabel("force [N]")
        ax5.set_title("External force")
        ax5.grid(True)

        # Position X
        ax6 = fig.add_subplot(4, 3, 8)
        ax6.plot(t, X, "b.-", label="X")
        ax6.set_ylabel("position [m]")
        ax6.set_ylim(np.min(X) * 0.99, np.max(X) * 1.01)
        ax6.set_title("Position and Lengths")
        ax6.legend()
        ax6.grid(True)

        # Lengths of MTC/CE/SEE
        ax7 = fig.add_subplot(4, 3, 11)
        ax7.plot(t, l_MTC, "g.-", label="l_MTC")
        ax7.plot(t, l_CE, "r.-", label="l_CE")
        ax7.plot(t, l_SEE, "y.-", label="l_SEE")
        ax7.set_xlabel("time [s]")
        ax7.set_ylabel("position [m]")
        ax7.set_ylim(np.min(np.concatenate([l_CE, l_SEE])) * 0.95, np.max(l_MTC) * 1.05)
        ax7.legend()
        ax7.grid(True)

        # Velocities
        ax8 = fig.add_subplot(4, 3, (9, 12))
        ax8.plot(t, V, "b.-", label="V_EXT")
        ax8.plot(t, v_MTC, "g.-", label="V_MTC")
        ax8.plot(t, v_CE, "r.-", label="V_CE")
        ax8.plot(t, v_SEE, "y.-", label="V_SEE")
        ax8.set_xlabel("time [s]")
        ax8.set_ylabel("velocity [m/s]")
        ax8.set_title("Velocities")
        ax8.legend()
        ax8.grid(True)

        # Small SEE force-elongation inset
        inset_rect = [0.45, 0.673, 0.102, 0.234] if not isoQ else [0.512, 0.734, 0.102, 0.171]
        inset_ax = fig.add_axes(inset_rect)
        mask_lin = f_MTC >= f_MTC_th
        mask_nl = ~mask_lin
        inset_ax.plot(1e3 * dx[mask_lin], 1e-3 * f_MTC[mask_lin], "gx-", label="linear")
        inset_ax.plot(1e3 * dx[mask_nl], 1e-3 * f_MTC[mask_nl], "ro-", label="non-linear")
        inset_ax.set_xlabel(r"$\Delta l_{SEE}$ [mm]")
        inset_ax.set_ylabel("f_MTC [kN]")
        inset_ax.set_title("SEE")
        inset_ax.grid(True)

        # Save figure
        fig_path_pdf = output_dir / f"{sim_id}.pdf"
        fig_path_png = output_dir / f"{sim_id}.png"
        fig.savefig(fig_path_pdf, bbox_inches="tight")
        fig.savefig(fig_path_png, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # A.9 Write txt output
    # ------------------------------------------------------------------
    if save_txt:
        txt_path = output_dir / f"{sim_id}.txt"
        with txt_path.open("w") as fid:
            fid.write(
                "This file contains data from a simulated leg extension "
                "using KneeExtElastic translated to Python.\n"
            )
            fid.write("ID:\t" + sim_id + "\n")
            fid.write("Date of simulation [yyyy-mm-dd-HH-MM-SS]:\t")
            fid.write(datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "\n")
            fid.write("#\n")

            # Options
            fid.write("SOLVER AND SOLVER OPTIONS\n")
            fid.write(f"Saved to:\t{output_dir}\n")
            fid.write(f"MaxStep [-]:\t{maxstep:.6g}\n")
            fid.write(f"RelTol [-]:\t{reltol:.6g}\n")
            fid.write("Refine [-]:\t1\n")
            fid.write(f"t0_sim [s]:\t{t0:.6g}\n")
            fid.write(f"te_sim [s]:\t{te:.6g}\n")
            fid.write(f"Iso:\t{int(isoQ)}\n")
            fid.write("#\n")

            # Environment and initial conditions
            fid.write("ENVIRONMENT AND INITIAL CONDITIONS\n")
            fid.write(f"Inclination angle [deg]:\t{phi:.6g}\n")
            fid.write(f"Constant of gravity [m/s^2]:\t{g_earth:.6g}\n")
            fid.write(f"Resulting gravity [m/s^2]:\t{g:.6g}\n")
            fid.write(f"Mass of sledge [kg]:\t{msledge:.6g}\n")
            fid.write(f"Additional load [kg]:\t{add:.6g}\n")
            fid.write(f"Total mass [kg]:\t{m:.6g}\n")
            fid.write(f"Initial Knee-Ext-Angle [deg]:\t{KA0:.6g}\n")
            fid.write(f"X0 [m]:\t{X0:.6g}\n")
            fid.write(f"V0 [m/s]:\t{V0:.6g}\n")
            fid.write(f"dV0 [m/s^2]:\t{dV0:.6g}\n")
            fid.write(f"Innervation t0 [s]:\t{t_on:.6g}\n")
            fid.write("#\n")

            # Subject parameters
            fid.write("SUBJECT-PARAMETERS\n")
            fid.write("LEG-PARAMETERS\n")
            fid.write(f"lo [m]:\t{lt:.6g}\n")
            fid.write(f"lu [m]:\t{ls:.6g}\n")
            fid.write(f"lr [m]:\t{lr:.6g}\n")
            fid.write(f"ku [m]:\t{ptl:.6g}\n")
            fid.write("#\n")

            fid.write("MUSCLE-TENDON\n")
            fid.write("*\n")
            fid.write("*ACTIVATION DYNAMICS\n")
            fid.write(f"U [1/s]:\t{U:.6g}\n")
            fid.write(f"Apre [1/s]:\t{Apre:.6g}\n")
            fid.write("*\n")
            fid.write("*FORCE-VELOCITY RELATION\n")
            fid.write(f"a [N]:\t{a:.6g}\n")
            fid.write(f"b [m/s]:\t{b:.6g}\n")
            fid.write(f"c [W]:\t{c:.6g}\n")
            fid.write("*\n")
            fid.write("*SERIAL ELASTIC ELEMENT\n")
            fid.write(f"k_see_lin [N/m]:\t{ksee:.9g}\n")
            fid.write(f"k_see_nonlin [N/m^n_see]:\t{ksnl:.9g}\n")
            fid.write(f"n_see_nonlin []:\t{ns:.9g}\n")
            fid.write(f"QT length = ku [m]:\t{ptl:.6g}\n")
            fid.write(f"lin_to_nonlin_TH [%]:\t{SEE_TH:.6g}\n")
            fid.write("*\n")
            fid.write("*PARALLEL ELASTIC ELEMENT\n")
            fid.write(f"k_pee [N/m^n_pee]:\t{kpee:.9g}\n")
            fid.write(f"n_pee []:\t{np_pee:.9g}\n")
            fid.write("*\n")
            fid.write("*FORCE-LENGTH RELATIONSHIP\n")
            fid.write(f"KA_l_CE_opt [deg]:\t{KA_opt:.6g}\n")
            fid.write(f"WIDTH []:\t{WIDTH:.6g}\n")
            fid.write(f"l_CE_opt [m]:\t{l_CE_opt:.6g}\n")
            fid.write("#\n")
            fid.write("CONVERTED FORCE-VELOCITY-PARAMETERS\n")
            fid.write(f"vmax [m/s]:\t{vmax:.6g}\n")
            fid.write(f"fiso [N]:\t{fiso:.6g}\n")
            fid.write(f"pmax [W]:\t{pmax:.6g}\n")
            fid.write(f"vopt [m/s]:\t{vopt:.6g}\n")
            fid.write(f"fopt [N]:\t{fopt:.6g}\n")
            fid.write(f"efficiency [-]:\t{eta:.6g}\n")
            fid.write(f"curvature [-]:\t{kappa:.6g}\n")
            fid.write("#\n")

            # Header and data
            header = [
                "Time [s]",
                "Fz [N]",
                "X [m]",
                "V [m/s]",
                "A [m/s^2]",
                "V_SEE [m/s]",
                "F_CE [N]",
                "V_CE [m/s]",
                "F_PEE [N]",
                "F_MTC [N]",
                "V_MTC [m/s]",
                "GeomFun [-]",
                "dGeomFun [-]",
                "ActFun [-]",
                "force-length []",
                "l_SEE [m]",
                "l_CE [m]",
                "l_MTC [m]",
            ]
            fid.write("\t".join(header) + "\n")

            data_matrix = np.column_stack(
                [
                    t,
                    F,
                    X,
                    V,
                    dV,
                    v_SEE,
                    f_CE,
                    v_CE,
                    f_PEE,
                    f_MTC,
                    v_MTC,
                    GX,
                    dG,
                    AT,
                    FL,
                    l_SEE,
                    l_CE,
                    l_MTC,
                ]
            )

            for row in data_matrix:
                fid.write("\t".join(f"{val:20.10g}" for val in row) + "\n")

    results = dict(
        t=t,
        F=F,
        X=X,
        V=V,
        dV=dV,
        v_SEE=v_SEE,
        f_CE=f_CE,
        v_CE=v_CE,
        f_PEE=f_PEE,
        f_MTC=f_MTC,
        v_MTC=v_MTC,
        GX=GX,
        AT=AT,
        FL=FL,
        l_SEE=l_SEE,
        l_CE=l_CE,
        l_MTC=l_MTC,
    )
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Simulate a leg extension with elastic elements "
        "(Python port of KneeExtElastic)."
    )
    parser.add_argument(
        "--isoQ",
        action="store_true",
        help="Simulate an isometric experiment (default: dynamic).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to store output files (txt, pdf, png).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not generate/save plots.",
    )
    parser.add_argument(
        "--no-txt",
        action="store_true",
        help="Do not write the txt result file.",
    )

    args = parser.parse_args()

    results = knee_ext_elastic(
        isoQ=args.isoQ,
        output_dir=args.output_dir,
        save_plots=not args.no_plots,
        save_txt=not args.no_txt,
    )

    print(
        f"Simulation finished. "
        f"Computed {len(results['t'])} time steps. "
        f"Output written to {args.output_dir!r}."
    )


if __name__ == "__main__":
    main()

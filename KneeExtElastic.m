%% A MAIN FUNCTION
function [ t,F,X,V,dV,v_SEE,f_CE,v_CE,f_PEE,f_MTC,v_MTC,GX,AT,...
    FL,l_SEE,l_CE,l_MTC ] = KneeExtElastic
% JUMPSIM_DISS_HP simulates a leg extension at an inclined legpress
%
%  It includes a parallel-elastic element,
%  a non-linear to linear serial elastic element, and
%  a force-length relation.
%  The Newtonian equation of motion is expressed as an ODE wich considers
%  initial conditions and neuromuscular properties provided by th user.
%
%  Finally tested with Matlab R2016b (Student License)
%  Required licenses:
%    - matlab
%
%  USAGE
%    Just run the file.
%    Change parameters in the secton "Input" if you like.
%
%  REFERENCE
%    H. Penasso and S. Thaller, ?Determination of Individual Knee-Extensor
%    Properties from Leg-Extensions and Parameter Identification,?
%    Mathematical and Computer Modelling of Dynamical Systems,
%    vol. in press, in press
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
%   [7] Haeufle, D. F. B., G?nther, M., Bayer, A., & Schmitt, S. (2014).
%      Hill-type muscle model with serial damping and eccentric
%      force-velocity relation. Journal of Biomechanics, 47(6), 1531?1536.
%      doi:10.1016/j.jbiomech.2014.02.009
%   [8] Pandy, M. G., Zajac, F. E., Sim, E., & Levine, W. S. (1990).
%      An optimal control model for maximum-height human jumping.
%      Journal of Biomechanics, 23(12), 1185?98.
%      Retrieved from http://www.ncbi.nlm.nih.gov/pubmed/2292598
%  
%  v.5.2 by Harald Penasso (16.11.2015) (cleaned up: 24.11.2016)
%% A.1 Input

  % Options
    t0   = 0;   % Start integration at this time [s]
    te   = 6.5; % Latest end of integration at this time [s]
                % (can be overruled by a solver stopping condition)
    reltol = 1e-8;   % ode45 option
    maxstep = 0.005; % ode45 option
    refine = 1;      % ode45 option
    path = {'/Users/haraldpenasso/Desktop/n/'}; % Path to final plots/results
    
  % Initial conditions and neuromuscular properties
    isoQ = 0;% true (1):  simulates an isometric experiment;
             % false (0): simulates a dynamic experiment.
               % WARNING: The initial load determines the initial force 
               % at the isometric condition. add = 40 is close to reality
    phi  = 90;  % Inclination of the plane where the point-mass moves [deg]
    g_earth = 9.81; % Constant of gravity [m/s^2]
    msledge = 32.5; % Partial mass #1 (mass of empty sledge: 32.5) [kg]
    add = 60;       % Partial mass #2 (additional weight: 40.0) [kg]
    KA0 = 120;   % Initial knee-extension-angel [deg]
    V0   = 0;    % Initial condition for ODE: first derivative of X [m/s]
    dV0  = 0;    % Initial condition for ODE:   2nd derivative of X [m/s^2]
    t_on = 0.25;  % Movement-initiation-time of actavation dynamics
    
  % Geometry
    lt  = 0.43;  % Length of the thigh [m]
    ls  = 0.43;  % Length of the shank [m]
    cir = 0.37;  % Circumference of the knee [m]
    ptl  = 0.08; % Patellar tendon length [m]
                 % Distance: middele patella to tuberositas tibiae
                 
  % Muscle-tendon complex
   % Activation dynamics
    U   = 12;       % Rate-constant of muscle-activation-dynamics [1/s]
    
   % Serial elastic element
    ksee = 1777000; % Linear stiffness of seiral elastic element [N/m]
    ns  = 1;        % n^th power of toe-region of the SEE (ksee x^ns)
    SEE_TH = 1;     % Transition point from nonlinear to linear SEE region
                    % Represents the perzentage (e[0,1]) of fiso
                    
   % Parallel elastic element
    kpee   = 0;     % Stiffness of parallel elastic element [N/m]
                      % if kp = 0, the PEE is inactive
    np   = 2;       % n^th power of parallel elastic element (kpee x^np)
    
   % Contractile element
    a    = 7.920643997366427e3; % Hill's parameter "a" [N]
    b    = 0.352028622105175;   % Hill's parameter "b" [m/s]
    c    = 9.124808590471668e3; % Hill's parameter "c" [W]
    
   % Force-length relationship
    KA_opt = 120;  % Knee-ext.-angle @ geometry_funs are at opt. length [1]
    WIDTH =  0.56; % Width of the force-length parabola: 0.56 [4]
                   % ... if set to inf the f-l relation is ineffective
    l_CE_opt = 0.09; % optimal geometry_fun-length  [m] [4,8]
    
%% A.2 Initial Calculations

   % Parameter convertions
    [U,a,b,c,vmax,fiso,pmax,vopt,fopt,eta,kappa]=params([U,a,b,c]);
    
   % Serial elastic element
    f_MTC_th = fiso * SEE_TH; % Force value at SEE lin to nonlin transition
    dx_th   = (ns*f_MTC_th)/ksee; % Change in SEE length at transition
    ksnl = ksee.*ns.^(-1).*dx_th.^(1+(-1).*ns); % Value for non-lnear part
    
   % Function of geometry
    lr = cir/(2*pi); % Approximated moment-arm of the knee [m]
    X0 = sqrt(lt^2+ls^2-2*lt*ptl*cos(KA0*pi/180)); % Init. position [m]
    g = g_earth*sin(phi*pi/180); % Gravity relative to inclination [m/s^2]
    m = msledge + add; % Total mass of sledge (Sum af mass#1 and #2) [kg]
    % Initial values
    [G_val0,~,dlmtc0] = geometry_fun(X0,V0,lt,ls,lr,ptl,KA_opt);
    
   % Muscle-tendon complex
    f_MTC0 = m*(dV0 + g)/G_val0; % initial force of the MTC
    
   % Set the initial region of SEE
    if f_MTC0 > f_MTC_th % linear
        dx0 = (f_MTC0+ksee*dx_th*(1-1/ns))/ksee;
    else                   % non-linear
        dx0 = nthroot(f_MTC0/ksnl,ns);
    end
    
   % Initial values of force-length relation and parallel elastic element
    [fL0,f_PEE0]=fl_fpee_lsee_fun(dx0,dlmtc0,ptl,l_CE_opt,WIDTH,kpee,np);
    
   % Initial value of activation dynamics
    Apre = (a.*b.*fL0.*G_val0+(-1).*c.*fL0.*G_val0).^(-1).*(b.*f_PEE0.*...
        G_val0+(-1).*b.*g.*m);
    
%% A.3 Filenames of output
    if add >= 99  % At tis line, it alters only the filename of the output;
        add = 99; % Adds information of mass#2 to the saved filename
    end      % WARNING: files using add > 99 will have identical filenames!
    id = {['psim00g',num2str(add),'v01']}; % Filename of output files

%% A.4 Check initial settings 
    if KA0 < 60
        error('The initial knee-extension-angle is to small (intAK < 60?)')
    end
    
    if a/fiso > 1
        error('The curvature aof the FV is > 1')
    end
    
    if pmax/c > 0.7
        error('The efficiency is to big (eta > 0.7)')
    end

    if pmax/c < 0.05
        error('The efficiency is small (eta < 0.05)')
    end
    
    if te <= t_on
        error('The contraction must begin within [t0,te) (te > t_on)')
    end

%% A.5 ODE solving
    opts=odeset('MaxStep',maxstep,'RelTol', reltol,'Events',@events,...
        'Refine',refine); % ode45 solver settings
    [t,x] = ode45(@model_fun,[t0,te],[X0,V0,dV0],opts); % ... Solve ODE
    
%% A.6 Results
    X  = x(:,1); % Position data [m]
    V  = x(:,2); % Velocity data [m]
    dV = x(:,3); % Acceleration data [m/s^2]

%% A.7 Calculate the values of sub-systems

  % External Force
    F = m*x(:,3)+m*g; % External Force [N]
    
  % Geometrical Function # d/dt Geometical Fun # Changes in length of MTC
    [GX,dG,dlmtc] = geometry_fun(X,V,lt,ls,lr,ptl,KA_opt);
    
  % Force of the muscle (CE+PEE) == force of the MTC == force of the SEE
    f_MTC = m*(dV + g)./GX;
    
  % Velocity of the MTC
    v_MTC = GX.*V;
    
  % Length change of the SEE
    dx = zeros(size(f_MTC)); % Pre-allocation only
    
   % Linear region
    dx(f_MTC>=f_MTC_th) = (f_MTC(f_MTC>f_MTC_th)+ksee.*dx_th.*...
        (1-1./ns))./ksee;
    
   % Non-linear region
    dx(f_MTC<f_MTC_th) = nthroot(abs(f_MTC(f_MTC<=f_MTC_th)./ksnl),ns);
    
  % Force-length relation # Force of PEE # length of SEE # length of CE
    [FL,f_PEE,l_SEE,l_CE]=fl_fpee_lsee_fun(dx,dlmtc,ptl,l_CE_opt,...
                                           WIDTH,kpee,np);
                                       
  % Force of the CE
    f_CE = (f_MTC-f_PEE);
    
  % Activation dynamics
    AT = ES_fun(t,t_on,Apre,U)';
    
  % Velocity of the CE == velocity of the PEE == velocity of the muscle
    v_CE = c./(((f_MTC-f_PEE)./(AT.*FL))+a)-b;
    
  % Velocity of the SEE
    v_SEE = v_MTC - v_CE;
    
  % Length of the MTC
    l_MTC = l_CE + l_SEE;
    
%% A.8 Final plots
    fig = figure('units','normalized','outerposition',[0 0 1 1]);
    
  % CE/PEE/MTC force vs. time
    subplot(4,3,[1,4])
    plot(t,f_CE,'or-')
    hold on
    plot(t,f_PEE,'om-')
    hold on
    plot(t,f_MTC,'og-')
    xlabel('time [s]')
    ylabel('force [N]')
    title('Internal forces')
    leg = legend('F_{CE}','F_{PEE}','F_{MTC}','Location','Best');
    set(leg,'FontSize',14);
    grid on
    
  % Geometry vs. time
    subplot(4,3,[2,5])
    plot(t,GX,'oc-')
    title('Function of geometry')
    xlabel('time [s]')
    ylabel('ratio [-]')
    grid on
    
  % Activation dynamics vs. time
    subplot(4,3,3)
    plot(t,AT,'oc-')
    xlabel('time [s]')
    ylabel('ratio [-]')
    ylim([0,1.1])
    title('Activation dynamics')
    grid on
    
  % Force-length dynamics vs. time
    subplot(4,3,6)
    plot(t,FL,'.k-')
    xlabel('time [s]')
    ylabel('ratio [-]')
    ylim([min(FL)*.99,1.01])
    leg = legend('fl','Location','Best');
    set(leg,'FontSize',14);
    title('Force-length dynamics')
    grid on
    
  % External force vs. time
    subplot(4,3,[7,10])
    plot(t,F,'ob-')
    xlabel('time [s]')
    ylabel('force [N]')
    title('External force')
    grid on
    
  % External position X vs. time
    subplot(4,3,8)
    plot(t,X,'ob-');
    ylabel('position [m]')
    ylim([min(X)*0.99,max(X)*1.01])
    title('Position and Lengths')
    leg = legend('X','Location','Best');
    set(leg,'FontSize',14);
    grid on
    
  % MTC-/CE-/SEE lengths vs. time
    subplot(4,3,11)
    plot(t,l_MTC,'og-')
    hold on
    plot(t,l_CE,'or-')
    hold on
    plot(t,l_SEE,'oy-')
    xlabel('time [s]')
    ylabel('position [m]')
    ylim([min([l_CE;l_SEE])*0.95,max(l_MTC)*1.05])
    leg = legend('l_{MTC}','l_{CE}','l_{SEE}','Location','Best');
    set(leg,'FontSize',14);
    grid on
    
  % External/MTC/CE/SEE velocities vs. time
    subplot(4,3,[9,12])
    plot(t,V,'ob-')
    hold on
    plot(t,v_MTC,'og-')
    hold on
    plot(t,v_CE,'or-')
    hold on
    plot(t,v_SEE,'oy-')
    xlabel('time [s]')
    ylabel('velocity [m/s]')
    title('Velocities')
    leg=legend('V_{EXT}','V_{MTC}','V_{CE}','V_{SEE}','Location','Best');
    set(leg,'FontSize',14);
    grid on
    
  % Make all text in the figure to size 14 and bold
    figureHandle = gcf;
    set(findall(figureHandle,'type','text'),'fontSize',18,...
        'fontWeight','bold')
    
  % Add additional force-elongation plot of the SEE 
    % Place second set of axes on same plot
    if ~isoQ
        handaxes2 = axes('Position', [0.45,0.673,0.102,0.234]);
    else
        handaxes2 = axes('Position', [0.512,0.734,0.102,0.171]);
    end
    plot(1e3*dx(f_MTC>=f_MTC_th),1e-3*f_MTC(f_MTC>=f_MTC_th),'xg-',...
        1e3*dx(f_MTC<f_MTC_th),1e-3*f_MTC(f_MTC<f_MTC_th),'or-')
    grid on
    set(handaxes2, 'Box', 'off')
    xlabel('\Delta l_{SEE} [mm]','FontSize',14)
    ylabel('f_{MTC} [kN]','FontSize',14)
    title('SEE')

  % Save figue as FIG and PDF 
    set(findall(fig,'type','text'),'fontSize',18,'fontWeight','bold')
    set(fig, 'PaperPosition', [0 0 48 38]);
    set(fig, 'PaperSize', [48 38]);
    saveas(fig,[path{:},id{:},'.fig'],'fig')
    saveas(fig,[path{:},id{:},'.pdf'],'pdf')
  
%% A.9 Write settings, initials, and ode45 results to txt-file

  % Header and data-matrix
    header = {'Time [s]','Fz [N]','X [m]', 'V [m/s]',...
        'A [m/s^2]','V_SEE [m/s]','F_CE [N]','V_CE [m/s]','F_PEE [N]',...
        'F_MTC [N]','V_MTC [m/s]','GeomFun [-]','dGeomFun [-]',...
        'ActFun [-]','force-length []','l_SEE [m]','l_CE [m]','l_MTC [m]'};
    data = [t,F,X,V,dV,v_SEE,f_CE,v_CE,f_PEE,f_MTC,v_MTC,GX,dG,AT,...
        FL,l_SEE,l_CE,l_MTC];
    
  % Open file
    fid = fopen([path{:},id{:},'.txt'],'w');
    
  % Write preamble
    fprintf(fid, '%s', ['This file contains data from a simulated leg'... 
            ' extension using ',mfilename,'.m (by Harald Penasso).']);
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'ID:');
    fprintf(fid, '%s', id{:});
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'Date of simulation [yyyy-mm-dd-HH-MM-SS]:');
    fprintf(fid, '%s', datestr(now,'yyyy-mm-dd-HH-MM-SS'));
    fprintf(fid, '\n');
    fprintf(fid, '%s', '#');
    
  % Options
    fprintf(fid, '\n');
    fprintf(fid, '%s', 'SOLVER AND SOLVER OPTIONS');
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'Saved to:');
    fprintf(fid, '%s', path{:});
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'MaxStep [-]:');
    fprintf(fid, '%s', num2str(maxstep,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'RelTol [-]:');
    fprintf(fid, '%s', num2str(reltol,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'Refine [-]:');
    fprintf(fid, '%s', num2str(refine,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 't0_sim [s]:');
    fprintf(fid, '%s', num2str(t0,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'te_sim [s]:');
    fprintf(fid, '%s', num2str(te,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'Iso:');
    fprintf(fid, '%s', num2str(isoQ,1));
    fprintf(fid, '\n');
    fprintf(fid, '%s', '#');
    
  % Initials
    fprintf(fid, '\n');
    fprintf(fid, '%s', 'ENVIRONMENT AND INITIAL CONDITIONS');
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'Inclination angle [deg]:');
    fprintf(fid, '%s', num2str(phi,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'Constant of gravity [m/s^2]:');
    fprintf(fid, '%s', num2str(g_earth,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'Resulting gravity [m/s^2]:');
    fprintf(fid, '%s', num2str(g,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'Mass of sledge [kg]:');
    fprintf(fid, '%s', num2str(msledge,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'Additional load [kg]:');
    fprintf(fid, '%s', num2str(add,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'Total mass [kg]:');
    fprintf(fid, '%s', num2str(m,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'Initial Knee-Ext-Angle [deg]:');
    fprintf(fid, '%s', num2str(KA0,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'X0 [m]:');
    fprintf(fid, '%s', num2str(X0,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'V0 [m/s]:');
    fprintf(fid, '%s', num2str(V0,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'dV0 [m/s^2]:');
    fprintf(fid, '%s', num2str(dV0,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'Innervation t0 [s]:');
    fprintf(fid, '%s', num2str(t_on,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s', '#');
    
  % Subject
    fprintf(fid, '\n');
    fprintf(fid, '%s', 'SUBJECT-PARAMETERS');
   % Function of geometry
    fprintf(fid, '\n');
    fprintf(fid, '%s', 'LEG-PARAMETERS');
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'lo [m]:');
    fprintf(fid, '%s', num2str(lt,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'lu [m]:');
    fprintf(fid, '%s', num2str(ls,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'lr [m]:');
    fprintf(fid, '%s', num2str(lr,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'ku [m]:');
    fprintf(fid, '%s', num2str(ptl,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s', '#');
   % Muscle
    fprintf(fid, '\n');
    fprintf(fid, '%s', 'MUSCLE-TENDON');
    fprintf(fid, '\n');
    fprintf(fid, '%s', '*');
    fprintf(fid, '\n');
    fprintf(fid, '%s', '*ACTIVATION DYNAMICS');
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'U [1/s]:');
    fprintf(fid, '%s', num2str(U,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'Apre [1/s]:');
    fprintf(fid, '%s', num2str(Apre,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s', '*');
    fprintf(fid, '\n');
    fprintf(fid, '%s', '*FORCE-VELOCITY RELATION');
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'a [N]:');
    fprintf(fid, '%s', num2str(a,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'b [m/s]:');
    fprintf(fid, '%s', num2str(b,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'c [W]:');
    fprintf(fid, '%s', num2str(c,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s', '*');
    fprintf(fid, '\n');
    fprintf(fid, '%s', '*SERIAL ELASTIC ELEMENT');
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'k_see_lin [N/m]:');
    fprintf(fid, '%s', num2str(ksee,9));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'k_see_nonlin [N/m^n_see]:');
    fprintf(fid, '%s', num2str(ksnl,9));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'n_see_nonlin []:');
    fprintf(fid, '%s', num2str(ns,9));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'QT length = ku [m]:');
    fprintf(fid, '%s', num2str(ptl,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'lin_to_nonlin_TH [%]:');
    fprintf(fid, '%s', num2str(SEE_TH,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s', '*');
    fprintf(fid, '\n');
    fprintf(fid, '%s', '*PARALLEL ELASTIC ELEMENT');
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'k_pee [N/m^n_pee]:');
    fprintf(fid, '%s', num2str(kpee,9));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'n_pee []:');
    fprintf(fid, '%s', num2str(np,9));
    fprintf(fid, '\n');
    fprintf(fid, '%s', '*');
    fprintf(fid, '\n');
    fprintf(fid, '%s', '*FORCE-LENGTH RELATIONSHIP');
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'KA_l_CE_opt [deg]:');
    fprintf(fid, '%s', num2str(KA_opt,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'WIDTH []:');
    fprintf(fid, '%s', num2str(WIDTH,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'l_CE_opt [m]:');
    fprintf(fid, '%s', num2str(l_CE_opt,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s', '#');
    fprintf(fid, '\n');
    fprintf(fid, '%s', 'CONVERTED FORCE-VELOCITY-PARAMETERS');
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'vmax [m/s]:');
    fprintf(fid, '%s', num2str(vmax,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'fiso [N]:');
    fprintf(fid, '%s', num2str(fiso,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'pmax [W]:');
    fprintf(fid, '%s', num2str(pmax,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'vopt [m/s]:');
    fprintf(fid, '%s', num2str(vopt,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'fopt [N]:');
    fprintf(fid, '%s', num2str(fopt,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'efficiency [-]:');
    fprintf(fid, '%s', num2str(eta,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', 'curvature [-]:');
    fprintf(fid, '%s', num2str(kappa,6));
    fprintf(fid, '\n');
    fprintf(fid, '%s', '#');
    
  % Header
    fprintf(fid, '\n');
    fprintf(fid, '%s\t', header{:});
    fprintf(fid, '\n');
    
  % Write data
    for i=1:length(data) % Write one column at each i-th iteration
        fprintf(fid, '%20.10g\t', data(i,:));
        fprintf(fid, '\n');
    end
    
  % Close file
    fclose(fid);
    
%% A.10 Final message
    disp(' ')
    disp(['Done. I saved the results to', path{:},id{:},'.txt'])
    disp(' ')

%% B NESTED-FUNCTIONS

%% B.1 THE MODEL

function [ X ] = model_fun(t,Y)
% MODEL_FUN contains the model-equation of the leg-extension task
%
%   This funcion is build in order to be send to an ode-solver
%
%    INPUT
%      t ... Time [s]
%      Y ... Y-value of the actual step to solve the next iteration step
%             1st column: Position [m]
%             2nd column: Velocity [m/s]
%             3nd column: Acceleration [m/s^2]
%
%    OUTPUT
%      X ... Numerical value of derivative of the ODE
%             1st column: Position [m]
%             2nd column: Velocity [m/s]
%             3nd column: Acceleration [m/s^2]

% Manipulations & error-checking before the contraction is initiatd (t_on)
    if t<=t_on % Time befor the contraction is initiated
       % Check for initial concentric movement:
        if geometry_fun(Y(1),Y(2),lt,ls,lr,ptl,KA_opt) * V0 > vmax
          error('Initial contraction-velocity > vmax');
        end
        
       % Check if the initial weight can be held
        if m*g > geometry_fun(X0, V0,lt,ls,lr,ptl,KA_opt) * (c/...
                (geometry_fun(X0, V0,lt,ls,lr,ptl,KA_opt)*V0 + b)-a)
          error('Subject cannot hold the weight at the initial position');
        end
    end
    
% Model-eqautions
  % MTC-force pependant changes of the SEE properties
    % Calculate value of function of geometry
    [GX,dG,dlmtc] = geometry_fun(Y(1),Y(2),lt,ls,lr,ptl,KA_opt);
    
    % Calculate muscle-tendon complex force
    f_MTC = abs(m*(Y(3) + g)/GX); % abs: Am Ende werden winzige negative
    % [ WARNING: abs() is needed to eliminate negligeable negative values
    %   before solver termination ]
    
  % Elongation of the SEE
    if f_MTC < f_MTC_th % non-linear part
        dx = nthroot(f_MTC/ksnl,ns);
    else % linear part
        dx = (f_MTC+ksee*dx_th*(1-1/ns))/ksee;
    end
    
  % Calculate values of force-length relation and PEE
    [FL,f_PEE] = fl_fpee_lsee_fun(dx,dlmtc,ptl,l_CE_opt,WIDTH,kpee,np);
    
  % Calculate value of activation dynamics
    AT = ES_fun(t,t_on,Apre,U);
    
  % Calculate actual position and velocity
  % Fix X if the isometric conditon is selected
    if isoQ
        X(1,1) = 0;
        X(2,1) = 0;
    else
        X(1,1) = Y(2);
        X(2,1) = Y(3);
    end
    
  % Calculate actual acceleration
    if f_MTC < f_MTC_th % non-linear part
        X(3,1) = GX.^(-1).*dG.*(g+Y(3))+(-1).*dx_th.^(1+(-1).*ns).*ksee...
            .*m.^(-1).*GX.*(dx).^((-1)+ns).*(a.*AT.*FL.*GX+(-1).*f_PEE.*...
            GX+m.*(g+Y(3))).^(-1).*(AT.*FL.*GX.*(a.*b+(-1).*c+a.*GX.*...
            Y(2))+(b+GX.*Y(2)).*((-1).*f_PEE.*GX+m.*(g+Y(3))));
    else % linear part
        X(3,1) = GX.^(-1).*(dG.*(g+Y(3))+ksee.*m.^(-1).*GX.^2.*((-1).*b+...
            GX.*((-1).*Y(2)+c.*AT.*FL.*(a.*AT.*FL.*GX+(-1).*f_PEE.*GX+...
            m.*(g+Y(3))).^(-1))));
    end

end % <<< function (model_fun) end

%% B.1.1 FUNCTION NESTED TO MAIN FUNCTION

% STOP INTEGRATION OF ODE45 AT EVENT DETECTED
function [value,isterminal,direction] = events(~,x)
% EVENTS locates events during the integration of the ODE and aborts the
% integration if the condition(s) are met. An "odeset(...)" function.

% Condition for abort is returned in "value" and is true if exists:
  %   01  negative acceleration
  %   02  knee-angle above 179 deg
  
   % Conditions 01 and 02
    value = [m*x(3)+m*g,...
             acos( (lt^2+ls^2-x(1)^2)/(2*lt*ls) )*180/pi-179];
    isterminal = [1,1]; % stop the integration
    direction  = [0,0];
    
end  % <<< nested function (events) end
% NESTED FUNCTION END #####################################################
%##########################################################################
end  % <<< main function (JumpSim_DISS_HP) end

%% C SUB-FUNCTIONS

%% C.1 FUNCT. OF GEOM. & FORCE-LENGTH RELATIONS & PARALLEL ELASTIC ELEM.
function [G,dG,dlMTC]=geometry_fun(X,dX,lt,ls,cir,ptl,KA_opt)
% GEOMETRY_FUN calculates geometrical relations between internal
% (MTC) and external force values. Credit: Prof. Kappel, jumpsens.m(!),
% which was extedned by the derivative of the geometrical realtion and the
% length-change of the muscle-tendon complex (MTC)
%
%    INPUT
%      X           ... Actual position data [m]
%     dX           ... Velocity data [m/s]
%      lt          ... Length of the thigh [m]
%      ls          ... Length of the shank [m]
%      cir         ... Moment arm of the knee [m]
%      ptl         ... Distance: middele patella to tuberositas tibiae [m]
%      KA_opt      ... Optimal knee-ext.-angle [deg] [1]
%                      (The angel where opt. fascicle length is reached)
%
%    OUTPUT
%      G           ... Ratio of gemetrical relation of the model-leg [-]
%      dG          ... d/dt GEOM [-]
%      dlMTC       ... Change of MTC rlative to its length at optimal
%                      knee-angle [m]

% Simplifications, re-naming
  ko = lt;
  r  = cir;
  
% Calculation
    [G,dG,dlMTC] = Gfuncvec(X);
    G(X>lt+ls) = 0;
    
%% C.1.1 NESTED FUNCTIONS FROM PROF. KAPPEL, JUMPSENS.M
  function [G,dG,dl_MTC]=Gfuncvec(x)
      
      % Calculate actual knee-angle
        sigma=acos((lt^2+ls^2-x.^2)/(2*lt*ls));
        
      % Calculate the half angle (beta) between model muscle and model
      % patellar-tendon direction at each knee-angle
        beta=betafunc(sigma);
        
      % Beta at optimal knee-angle, where it is assumed that the fascicle
      % length is optimal length too
        b_opt = betafunc(KA_opt*pi/180);
        
      % Length of the model muscle-tendon-complex @ 120 deg Knee-Ext.-A [1]
        l_MTC_opt = sqrt(lt^2+r^2-2*lt*r*cos(pi-b_opt-...
            asin((r*sin(b_opt))/lt)));
        
      % Actual length of the MTC
        l_MTC = sqrt(lt^2+r^2-2*lt*r*cos(pi-beta-asin((r*sin(beta))/lt)));
        
      % Change of model-muscle length due to knee extension
        dl_MTC = l_MTC_opt-l_MTC;
        
      % 1st derivative of the function of geometry
        dG = dG_fun(x,dX,sigma,beta);
        
      % Final value of the function of geometry
        G=r*x.*sin(beta)./(lt*ls*sin(sigma));
        
       %% C.1.1.1 NESTED #1
       function beta=betafunc(sigma)
           
           s = sigma; M = length(s); beta = zeros(M,1);
           
           for k=1:M
               rho=s(k);
               options=optimset('MaxFunEvals',12000,'MaxIter',10000);
               beta(k)=fminbnd(@bfunc,0,pi,options);
           end
           
           %% C.1.1.1.1 NESTED #2
           function w=bfunc(v) % v = beta (w = alpha)
               
               w=(2*v+asin(r*sin(v)/ko)+asin(r*sin(v)/ptl)-rho)^2;
               
           end % <<< function (bfunc) end
           
       end % <<< function (betafunc) end
       
  %% C.1.2 FUNCTION TO CALC. THE 1ST DERIVATIVE OF THE FUNCTION OF GEOMETRY
  function [ dG ] = dG_fun(X,dX,alpha,beta)
  % DG_FUN calculates the 1st derivative of the geometry function,
  % including the derivative of the empiric ratio
  %
  %  INPUT:
  %     X    ... Position data [m]
  %    dX    ... Velocity data [m/s]
  %    alpha ... Knee-angle [rad]
  %    beta  ... Angle at the patella [rad]
  %          +++ Uses anthropometric lengths from its parent-function
  %
  %  OUTPUT
  %    dG    ... Value of the 1st derivative of the geometrical function
    
    % These variables are set to disablede a functionality that was removed
    % from the version
      Ratio  = 1;
      dRatio = 0;
    
    % Derivative of the function of geometry
      dG = (lt.^(-1).*cir.*ls.^(-1).*csc(alpha).*(sin(beta).*X.*dRatio...
          +sin(beta).*Ratio.*dX+2.*lt.^(-1).*ls.^(-1).*((-1).*...
          cot(alpha).*sin(beta)+cos(beta).*(2+cos(beta).*...
          (ko.*cir.*(ko.^2+(-1).*cir.^2.*sin(beta).^2).^(-1).*...
          (1+(-1).*ko.^(-2).*cir.^2.*sin(beta).^2).^(1/2)+ptl.*cir.*...
          (ptl.^2+(-1).*cir.^2.*sin(beta).^2).^(-1).*(1+(-1).*ptl.^...
          (-2).*cir.^2.*sin(beta).^2).^(1/2))).^(-1)).*Ratio.*X.^2.*...
          ((-1).*lt.^(-2).*ls.^(-2).*((lt.^2+(-1).*ls.^2).^2+(-2).*...
          (lt.^2+ls.^2).*X.^2+X.^4)).^(-1/2).*dX));
      
  end % <<< function (dG_fun) end
  
 end % <<< function (Gfunvec) end

end % <<< function (geometry_fun) end


%% C.2 FUNCTION OF FORCE-LENGTH RELATIONSHIP & PEE
function [ fl, f_PEE, l_SEE, l_CE ] = ...
    fl_fpee_lsee_fun(dx,dlmtc,QPTL,l_CE_opt,WIDTH,k_pee,n_pee)
% FL_FUN  calculates a ratio of the force-length relationship based
% on [4]. According to [4,8] the optimal fascicel-length is 0.09 m and
% according to [1] is reached at 120 ? knee-extension-angle.
%
%  INPUT:
%   dx       ... length-change of the SEE [m]
%   dlmtc    ... length-change of the MTC relative to its length @ optimal 
%                knee-flexion-angle [1] [m]
%   QPTL     ... length of the merged quadriceps-to-tibia tendon [m]
%   l_CE_opt ... optimal length of the CE [4,8] [m]
%   WIDTH    ... Width of the force-length parabola [m] [4]
%   k_pee    ... length-change of the SEE [N/m^pee]
%   n_pee    ... n-th power of the PEE [-]
%
%  OUTPUT:
%   fl       ... ratio of the force-length relationship [-]
%   f_PEE    ... force produced by the parallel elastic element  [N]
%   l_SEE    ... length of the serial elastic element  [m]
%   l_CE     ... lengh of the contractile element [m]
  
  % Actual length of the SEE
    l_SEE = QPTL+dx;
    
  % Actual length of the CE
    l_CE = l_CE_opt-dx-dlmtc;
    
  % Calculate the force-length relation [4]
    if min(l_CE)>=(1-WIDTH)*l_CE_opt && min(l_CE)<=(1+WIDTH)*l_CE_opt
        c = -1/WIDTH^2;
        fl = c*(l_CE./l_CE_opt).^2-2*c*(l_CE./l_CE_opt)+c+1;
    else
        error('The CE is out of bounds!')
    end
    
  % Calculate the force of the parallel elsatic element: lpee == lce [6,7]
     f_PEE = (k_pee.*(l_CE-l_CE_opt).^n_pee);
     f_PEE(l_CE<l_CE_opt) = 0;
     
end % <<< function (fl_fpee_lsee_fun) end


%% C.3 FUNCTION OF ACTIVATION DYNAMICS
function [ A ] = ES_fun(t, t_on, Apre, U )
% ES_FUN  calculates the activation dynamics of the muscle
%
%   INPUT:
%     t    ... time vector [s]
%     t_on ... time when the contraction starts [s]
%     Apre ... constant pre-activation level [1/s]
%     U    ... property of muscle activation [1/s]
%
%   OUTPUT:
%     A    ... ratio [0,1] of muscle activation dynamics

% Simplifications (to preserve full functionality)
    nmax = 1;
    umax = U;
    
% Calculate it
    A = zeros(1,length(t));
    A(t<t_on) = Apre;
    A(t>=t_on)  =  exp(1).^((-1).*exp(1).^((-1).*(t(t>=t_on)-t_on).*U).*...
        U.^(-1).*((-1).*Apre+umax+exp(1).^((t(t>=t_on)-t_on).*U).*...
        (Apre+((-1)+(t(t>=t_on)-t_on).*U).*umax))).*(Apre+(-1).*nmax)+nmax;
    
end % <<< function (ES_fun) end


%% C.4 PROPERTY CONVERSION FUNCTION
function[U,a,b,c,vmax,fiso,pmax,vopt,fopt,eta,kappa]=params(Par)
% PARAMS is a convenient muscle-properties conversion function
%
%   INPUT
%    Par ... vector containing the properties [U,a,b,c]
%
%   OUTPUT
%    returns the complete set of parameters based on the input provided

    U       = Par(1);
    a       = Par(2);
    b       = Par(3);
    c       = Par(4);
    vmax    = c/a-b;
    fiso    = c/b-a;
    pmax    = a*b+c-2*sqrt(a*b*c);
    vopt    = sqrt((b*c)/a)-b;
    fopt    = sqrt((a*c)/b)-a;
    eta     = pmax/c;
    kappa   = a/fiso;
    
end % <<< function (params) end

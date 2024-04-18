%%% PROGRAM TO INTEGRATE SAVANNA EQUATIONS IN SPACE (PDE)
% FROM BAUDENA AND RIETKERK 2013 THEORETICAL ECOLOGY
% 

%--------%%--------%%--------%%
%  PARAMETERS INITIALIZATION

clear all


global ct cg g1 g2 m1 m2 Dg Dw Dh L eta aa k w0 E bs ll R 
global Nx Ny dx dy dt fK2D

% ll=lambda; gg=gamma aa=alfa

ct = 6.1;   % g mm-1 m-2
cg = 7;    % g mm-1 m-2
g1 = 0.1;   % m2 g-1 d-1 (same as gSmax but divided by a height (for water)=1 mm
g2 = 0.1;    % m2 g-1 d-1
m1 = 0.2  ; % d-1
m2 = 0.2 ;  % d-1
Dg=0.1;
Dw = 0.1;    % m2 d-1
Dh = 20;     % m2 d-1 mm-1 (same as Yolanda's Dh but divided by a height h)=1 mm
L  = 20;     % m
eta= 0.1;      % d-1 (from 10-4 to 1)
%etav=[10^(-4) 0.001 0.01 0.1 1 5]; % mm-1 d-1 (same as Yolanda's Rc*f*gs but divided by a height h)=1 mm
aa = 0.2;    % d-1 
k  = 5;      % g m-2
w0 = 0.2;
%w0v=[0 0.4 0.8 1];
E  = 2.;    % d-1
%bs = 20.;    % g m-2 (value ??????)
bs=10;
ll = 0.2;    %

%R=0.9;      % m d-1
%Rv=[0.45 0.55 0.65 0.75]; % mm d-1 
Rv=[0.55 0.75];

% frac1=0.3;  % initial fraction of soil with trees
% frac2=0.3;    % initial fraction of soil with grass
% frac3=0.1;   % initial fraction of soil with tree seedlings

b10=15;        % Initial tree biomass  (g.m-2)
b20=15;        % Initial grass biomass  (g.m-2)

% SIMULATION PARAMETERS
N0year=10;
%T=365*Nyear   % final time (d)
dt=0.002;      % d

epsilon=10^(-9);

% SPATIAL DOMAIN PARAMETERS
LengthX  =     128*1;     % Length of the landscape in the X dimension
%LengthY  =     200;    % Length of the landscape in the Y dimension
m        =     64;     % Gridsize in a square grid

Nx=m; Ny=m;

dx= LengthX/Nx; dy=dx;
%dy= LengthY/Ny;

% DISPERSAL KERNEL

centro=zeros(m,m); centro(m/2,m/2)=1; centro((m/2+1),m/2)=1;
centro((m/2),(m/2)+1)=1; centro((m/2)+1,(m/2)+1)=1;
dist=dx*bwdist(centro);             % (m)
K2D=exp(-2.*dist./L);               % negative exponential kernel
K2D=K2D./(sum(sum(K2D)));           % normalizing to 1 (instead of 2/(pi*L2)
fK2D=fft2(K2D);                     % fft of the kernel

clear centro dist K2D

% RANDOM MATRIX FOR VEGETATION FIELD INITIALIZATION
% HIGH TREES HIGH GRASSES INITIAL CONDITION:

frac1=0.3;    % initial fraction of soil with trees
frac2=0.3;    % initial fraction of soil with grass

urandHTHG=zeros(Nx,Ny,2);
rand('twister',10132)
rand1=rand(Nx,Ny);
mi1=rand1<frac1;
urandHTHG(:,:,1)=urandHTHG(:,:,1)+mi1*b10; %Initial tree biomass  (g.m-2)
rand1(mi1)=9999;
mi2=(rand1<frac2+frac1);
urandHTHG(:,:,2)=urandHTHG(:,:,2)+mi2*b20; %Initial grass biomass (g m-2)


% LOW TREES HIGH GRASSES INITIAL CONDITION:

frac1=0.01;    % initial fraction of soil with trees
% NB SHOULD TRY WITH HIGHER FRAC2 E.G. 0.6!
frac2=0.6;    % initial fraction of soil with grass


urandLTHG=zeros(Nx,Ny,2);
rand('twister',10132)
rand1=rand(Nx,Ny);
mi1=rand1<frac1;
urandLTHG(:,:,1)=urandLTHG(:,:,1)+mi1*b10/100; %Initial tree biomass  (g.m-2)
rand1(mi1)=9999;
mi2=(rand1<frac2+frac1);
urandLTHG(:,:,2)=urandLTHG(:,:,2)+mi2*b20; %Initial grass biomass (g m-2)



% HIGH TREES LOW GRASSES INITIAL CONDITION:

frac1=0.6;      % initial fraction of soil with trees
frac2=0.01;     % initial fraction of soil with grass

urandHTLG=zeros(Nx,Ny,2);
rand('twister',10132)
rand1=rand(Nx,Ny);
mi1=rand1<frac1;
urandHTLG(:,:,1)=urandHTLG(:,:,1)+mi1*b10; %Initial tree biomass  (g.m-2)
rand1(mi1)=9999;
mi2=(rand1<frac2+frac1);
urandHTLG(:,:,2)=urandHTLG(:,:,2)+mi2*b20/100; %Initial grass biomass (g m-2)



% LOW TREES AND LOW GRASSES IN INITIAL CONDITION:

frac1=0.01;      % initial fraction of soil with trees
frac2=0.01;      % initial fraction of soil with grass

urandLTLG=zeros(Nx,Ny,2);
rand('twister',10132)
rand1=rand(Nx,Ny);
mi1=rand1<frac1;
urandLTLG(:,:,1)=urandLTLG(:,:,1)+mi1*b10/10; %Initial tree biomass  (g.m-2)
rand1(mi1)=9999;
mi2=(rand1<frac2+frac1);
urandLTLG(:,:,2)=urandLTLG(:,:,2)+mi2*b20/10; %Initial grass biomass (g m-2)


clear mi* rand1


% LOOPS OVER PARAM VALUES
for init=2:4
%     if init==1
%         urand=urandHTHG; clear urandHTHG
%         Nyear=N0year*3;
%    elseif init==2
    if init==2
        urand=urandLTHG; clear urandLTHG
        Nyear=N0year;
    elseif init==3
        urand=urandHTLG; clear urandHTLG
        Nyear=N0year;
    else
        urand=urandLTLG; clear urandLTLG
        Nyear=N0year;
    end
    for ii=1:length(Rv)
        if init==2 & ii==1
            continue
        end
        R=Rv(ii)
        tic
        % VARIABLE INTIAL CONDITIONS
        u=zeros(Nx,Ny,4);

        u(:,:,1:2)=urand;
        u(:,:,3)=R/(E+0.000001);
        u(:,:,4)=R/(0.2*w0+0.000001);


        % INITIALIZATION OF MATRIX WITH 4 TIME STEP - USEFUL FOR ADAM-BASHFORT 3-4
        % PREDICTOR CORRECTOR

        Fu=zeros(Nx,Ny,4,4);
        up=u;

        %%% STARTING TIME LOOP

        time=dt;

        % integrate with Euler for the first 3 time steps
        for i=1:3
            %while time<=T
            % EULER
            u=up;
            Fu(:,:,:,i)=rhs(time,u);
            up=u+dt*squeeze(Fu(:,:,:,i));
            %    up=u+dt.*rhs(time,u);
            time=time+dt;
        end

        %%%%%%%% YEAR=1
        iy=1;
        time=0;
        while time<=365
            %   Adams-Bamshfort predictor-corrector method:
            %   predictor
            up=predictor(u,Fu);
            %   evaluate rhs at n+1 (using up)
            Fu(:,:,:,4)=rhs(time,up);
            %   corrector
            up=corrector(u,Fu);
            %   recalculating rhs at n+1 (using u)
            Fu(:,:,:,4)=rhs(time,up);
            %   derivative shift for next iteration
            Fu(:,:,:,1:3)=Fu(:,:,:,2:4);

            u=up;
            time=time+dt;
              
            if (time>100 & time<=100+dt)|(time>200 & time<=200+dt)
                TT=time;
                eval(['save r',num2str(R),'init',int2str(init),'laplh2_',num2str(TT/365),'LOW1SMALL.mat ' ...
                    'u ct cg g1 g2 m1 m2 Dg Dw Dh L eta aa k w0 E bs ll R TT'])
            end
        end
        TT=(iy-1)*365+time
        eval(['save r',num2str(R),'init',int2str(init),'laplh2_',num2str(TT/365),'LOW1SMALL.mat ' ...
            'u ct cg g1 g2 m1 m2 Dg Dw Dh L eta aa k w0 E bs ll R TT'])

        % checking if u is lower than epsilon
        if u(:,:,1)<epsilon | u(:,:,2)<epsilon
            eval(['!mv r',num2str(R),'init',int2str(init),'laplh2_',num2str(TT/365),'LOW1SMALL.mat '...
                r',num2str(R),'init',int2str(init),'laplh2_finalLOW1SMALL.mat '])
            break
        end

        % checking if u has nan
        if max(max(isnan(u(:,:,1))>0)) | max(max(isnan(u(:,:,2))>0))
            break
        end


        for iy=2:Nyear
            time=0;
            while time<=365
                %   Adams-Bamshfort predictor-corrector method:
                %   predictor
                up=predictor(u,Fu);
                %   evaluate rhs at n+1 (using up)
                Fu(:,:,:,4)=rhs(time,up);
                %   corrector
                up=corrector(u,Fu);
                %   recalculating rhs at n+1 (using u)
                Fu(:,:,:,4)=rhs(time,up);
                %   derivative shift for next iteration
                Fu(:,:,:,1:3)=Fu(:,:,:,2:4);

                u=up;
                time=time+dt;
                %              TT=(iy-1)*365+time
            end
            TT=(iy-1)*365+time
            eval(['save r',num2str(R),'init',int2str(init),'laplh2_',num2str(TT/365),'LOW1SMALL.mat ' ...
                'u ct cg g1 g2 m1 m2 Dg Dw Dh L eta aa k w0 E bs ll R TT'])

            % checking if u is lower than epsilon
            if u(:,:,1)<epsilon | u(:,:,2)<epsilon
                eval(['!mv r',num2str(R),'init',int2str(init),'laplh2_',num2str(TT/365),'LOW1SMALL.mat '...
                    'r',num2str(R),'init',int2str(init),'laplh2_finalLOW1SMALL.mat '])
                break
            end

            % checking if u has nan
            if max(max(isnan(u(:,:,1))>0)) | max(max(isnan(u(:,:,2))>0))
                break
            end
        end
        
        if iy==Nyear
            eval(['!mv r',num2str(R),'init',int2str(init),'laplh2_',num2str(TT/365),'LOW1SMALL.mat '...
                'r',num2str(R),'init',int2str(init),'laplh2_finalLOW1SMALL.mat '])
        end
        toc
    end
end
exit
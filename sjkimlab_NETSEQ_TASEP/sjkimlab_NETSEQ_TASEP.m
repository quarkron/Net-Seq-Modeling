function output= NETSEQ_TASEP_function(input_parameters)
%--------------------------------------------------------------------------------------------------
% Code written by: Albur Hassan
% sjkimlab at University of Illinois at Urbana Champaign.
% Original creation date: 2026/01/24
% Last edited: 2026/01/25
% Contact: ahassan4@illinois.edu
%--------------------------------------------------------------------------------------------------
% Program Description: this function is a offbranch version of the albur_TASEP.m
% Notably this function does not include many features such as RNA degradation multiple ribosomes
% and is missing many different outputs. The purpose of this function is to serve as a speedier
% simulation to investigate NETSEQ output and hence only have features relevant to produce NETSEQ
% signals.
%--------------------------------------------------------------------------------------------------
% to run this program please call the function as NETSEQ_TASEP_function(input_parameters)
% where input_parameters is a struct object containing the parameters for the simulation
% An example of calling this program would be as follows:
%	parameters= struct;
%	parameters.KRutLoading=0.13;
%	parameters.RNAP_dwellTimeProfile= ones(1000,1);
%	final_output = sjkimlab_NETSEQ_TASEP(parameters);
%	for i =1:300
%		output = sjkimlab_NETSEQ_TASEP(parameters);
%		final_output.NETseq = final_output.NETseq+ output.NETseq;
%	end
%	final_output.NETseq= final_output.NETseq./300;
%--------------------------------------------------------------------------------------------------

	%% ── Default parameters ──────────────────────────────────────────────
	parameters =struct;
	parameters.RNAPSpeed =19;          % RNAP elongation speed (nt/s)
	parameters.ribospeed= 19;          % ribosome elongation speed (nt/s)
	parameters.kLoading =1/20;         % RNAP initiation rate (1/s)
	parameters.kRiboLoading = 0;       % ribosome loading rate (1/s)
	parameters.KRutLoading =0.13;      % Rho/rut loading parameter
	parameters.simtime=2000;           % total simulation time (s)
	parameters.glutime=1600;           % end of active transcription (s) — "glucose time"
	parameters.geneLength=3075;        % DNA length (bp), overridden by dwell profile length
	parameters.RNAP_dwellTimeProfile = ones(parameters.geneLength,1);

	% Override defaults with any user-supplied parameters
	if nargin >=1
		parameterlabels= fieldnames(input_parameters);
		for i = 1:numel(parameterlabels)
				parameters.(parameterlabels{i})= input_parameters.(parameterlabels{i});
		end
	end
	parameters.rutSpeed=5*parameters.ribospeed;     % Rho translocation speed (5x ribosome)
	parameters.geneLength= length(parameters.RNAP_dwellTimeProfile); % use profile length

	%% ── Unpack parameters into local variables ──────────────────────────
	RNAPSpeed=parameters.RNAPSpeed;
	riboSpeed = parameters.ribospeed;
	geneLength=parameters.geneLength;
	RNAP_width=35;    % footprint of RNAP on DNA (bp)
	dx=1;             % spatial step (bp)
	dt=0.1;           % time step (s)
	simtime=parameters.simtime;
	glutime=parameters.glutime;
	kLoading =parameters.kLoading;
	kRiboLoading = parameters.kRiboLoading;
	boolRNAPRiboCoupling=1;  % enable RNAP-ribosome coupling on collision

	%% ── Per-RNAP state arrays (grow as new RNAPs load) ──────────────────
	riboExitTimes =zeros(geneLength/dx,1);   % ribosome exit time at each position per RNAP
	Ribo_locs=[];                             % current position of each ribosome
	RNAP_RiboCoupling =[0];                   % 1 if RNAP-ribosome are coupled, 0 if not
	Ribo_width =30;                           % ribosome footprint (bp)
	rho_width =30;                            % Rho factor footprint (bp)

	%% ── Rho / premature termination setup ───────────────────────────────
	rut_sites =[round(500*geneLength/3075)];  % rut site positions (scaled to gene length)
	rutSpeed=parameters.rutSpeed;
	minRholoadRNA = 80-rho_width;             % min exposed RNA length for Rho loading
	rho_locs =[];                             % current Rho factor position per RNAP
	rut_loadT=[];                             % scheduled Rho loading time per RNAP per rut site
	rut_site=[1500];
	specificDwelltimeRho=dx/rutSpeed* ones(geneLength/dx,1);  % Rho dwell time per position
	tempRho=0;
	r_loc_time=zeros(2,length(rut_sites),2);
	PTpercent =0;
	PT_Model=2;   % Model 2: Rho loading prob ~ exposed nascent RNA length
	if PT_Model==1 ||PT_Model==2
		KRutLoading=0;
		PTpercent =parameters.KRutLoading;
	end

	%% ── Dwell-time profiles and initial scheduling ──────────────────────
	tempExitTimes =[];
	avgDwelltime1 = dx/RNAPSpeed;             % mean RNAP dwell time per nt (s/nt)
	riboavgDwelltime = dx/riboSpeed;          % mean ribosome dwell time per nt (s/nt)
	loadt= exprnd(1/kLoading);               % schedule first RNAP loading time
	rnap_locs=[];                             % current position of each RNAP
	Riboloadt= loadt + exprnd(1/kRiboLoading); % schedule first ribosome loading time
	specificDwelltime1 = avgDwelltime1 .* parameters.RNAP_dwellTimeProfile; % position-dependent RNAP dwell
	RibospecificDwelltime1=riboavgDwelltime*ones(geneLength/dx,1);         % uniform ribosome dwell
	RNAP_exitTimes = zeros(geneLength/dx,1);  % RNAP exit time at each position per RNAP


	%% ════════════════════════════════════════════════════════════════════
	%  MAIN SIMULATION LOOP
	%  Two phases:
	%    Active transcription (0 → glutime): RNAPs load and elongate
	%    Runoff (glutime → simtime): only elongation, no new loading
	%% ════════════════════════════════════════════════════════════════════
	for t = 0:dt:simtime

		%% ── RNAP loading (active phase only, t < glutime) ───────────────
		% If promoter is blocked (last RNAP too close), reschedule loading
		if(loadt <=t & t <glutime & ~isempty(rnap_locs) & rnap_locs(length(rnap_locs))-RNAP_width <=0 )
			loadt = t + exprnd(1/kLoading);
		end

		% If promoter is clear, load a new RNAP at position 1
		if(loadt <=t & t <glutime &(isempty(rnap_locs) ||rnap_locs(length(rnap_locs))-RNAP_width >=0 ) )
			rnap_locs(length(rnap_locs)+1) = 1;
			RNAP_exitTimes(:,length(rnap_locs)) = zeros(geneLength,1);
			RNAP_RiboCoupling(length(RNAP_RiboCoupling)+1) = 0;
			loadt = t + exprnd(1/kLoading);
			Riboloadt(length(rnap_locs)) = t + exprnd(1/kRiboLoading);
			Ribo_locs(length(rnap_locs),1)=0;
			rho_locs(length(rnap_locs),1)=0;
			riboExitTimes(:,length(rnap_locs),1)=zeros(geneLength,1);
			for rs_idx =1:size(rut_sites,2)
				rho_locs(length(rnap_locs)) =0;
				rut_loadT(length(rnap_locs),1:length(rut_site)) = simtime +1;
			end
		end

		%% ── RNAP elongation (runs in BOTH active and runoff phases) ─────
		for rnap = 1:length(rnap_locs)
			currentRNAPloc =rnap_locs(rnap);
			if rnap_locs(rnap) <=geneLength
				% Sample exit times for a window of bases ahead of this RNAP
				bases_evaluated =ceil(RNAPSpeed*10*dt);
				if rnap_locs(rnap)+bases_evaluated <=geneLength
					tempExitTimes = t+cumsum(exprnd(specificDwelltime1(currentRNAPloc:currentRNAPloc+bases_evaluated)));
				else
					tempExitTimes = t+cumsum(exprnd(specificDwelltime1(rnap_locs(rnap):geneLength)));
				end

				% Keep only exits that fall within this time step [t, t+dt]
				tempRNAP_exitTimes=  tempExitTimes((tempExitTimes>=t & tempExitTimes<=t+dt));

				%% ── Collision check with the RNAP ahead ─────────────────
				if rnap>1
					PrevRNAPloc =rnap_locs(rnap-1);
					% Skip over any prematurely terminated RNAPs (sentinel = geneLength+10)
					if PrevRNAPloc == geneLength + 10
						j=1;
						while j <= size(rnap_locs(1:rnap-1)) & rnap_locs(rnap-j) == geneLength + 10 &rnap-j >1;
							j =j+1;
						end
						if j == rnap || rnap-j <1
							PrevRNAPloc =geneLength+1;    % no active RNAP ahead
						else
							PrevRNAPloc = rnap_locs(rnap-j);
						end
					end
					% Positive overlap → collision with RNAP ahead
					overlap = (rnap_locs(rnap)+size(tempRNAP_exitTimes,1)-1)-PrevRNAPloc +RNAP_width;

					if PrevRNAPloc >= geneLength
						overlap =0;  % RNAP ahead already finished, no collision possible
                    end


					if overlap <=0
						% No collision: record exit times and advance
						RNAP_exitTimes(rnap_locs(rnap):(rnap_locs(rnap)+size(tempRNAP_exitTimes,1)-1),rnap)=tempRNAP_exitTimes;
						rnap_locs(rnap) = rnap_locs(rnap)+size(tempRNAP_exitTimes,1);
					else
						% Collision: advance only up to the point of contact
						RNAP_exitTimes(rnap_locs(rnap):(rnap_locs(rnap)+size(tempRNAP_exitTimes,1)-1-overlap),rnap)=tempRNAP_exitTimes(1:size(tempRNAP_exitTimes,1)-overlap);
						rnap_locs(rnap) = rnap_locs(rnap)+size(tempRNAP_exitTimes,1)-overlap;
					end
                else
					% First RNAP (no one ahead): always advance freely
					tempRNAP_exitTimes=  tempExitTimes((tempExitTimes>=t & tempExitTimes<=t+dt));
					RNAP_exitTimes(rnap_locs(rnap):(rnap_locs(rnap)+size(tempRNAP_exitTimes,1)-1),rnap)=tempRNAP_exitTimes;
					rnap_locs(rnap) = rnap_locs(rnap)+size(tempRNAP_exitTimes,1);
				end
			end

			%% ── Ribosome loading onto this RNAP's mRNA ──────────────────
			% Load ribosome once RNAP has moved far enough (>= Ribo_width bp)
			if(Riboloadt(rnap) <=t & rnap_locs(rnap) >=Ribo_width)
				Riboloadt(rnap) = simtime;   % only one ribosome per RNAP
				if size(Ribo_locs)==0
					Ribo_locs(1)=0;
					riboExitTimes(1,1) =0;
					Ribo_locs(rnap) = 1;
					riboExitTimes(1,rnap)= t;
				elseif size(Ribo_locs,1) <rnap
					Ribo_locs(rnap)=1;
					riboExitTimes(1,rnap) =t;
				else
					Ribo_locs(rnap)=1;
					riboExitTimes(:,rnap) = zeros(geneLength,1);
				end
            end


		end

		%% ── Rho factor / premature termination ──────────────────────────
		for RNA = 1:length(rnap_locs)

			% PT Model 2: Rho loading probability ∝ exposed (unprotected) RNA
			% Exposed RNA = RNAP pos - RNAP footprint - Rho footprint - ribosome pos
			if PT_Model==2 && rnap_locs(RNA)<geneLength
				PTRNAsize = rnap_locs(RNA)- RNAP_width-rho_width;
				PTRNAsize= PTRNAsize - Ribo_locs(RNA,1);
				if PTRNAsize>minRholoadRNA && 100*dt*rand <= PTpercent*PTRNAsize/geneLength
					% Place Rho randomly on the exposed RNA region
					temp_rho_loading_loc = rnap_locs(RNA)- RNAP_width- floor(rand*PTRNAsize);
					if temp_rho_loading_loc>rho_locs(RNA,1)
						rho_locs(RNA,1)=temp_rho_loading_loc;
					end
				end
			end

			for rs_idx =1:size(rut_sites,1)
				rut_site= rut_sites(rs_idx);

				% PT Model 1: percentage-based termination at rut sites
				if  PT_Model==1 && RNAP_exitTimes(rut_site,RNA) <t+dt && RNAP_exitTimes(rut_site,RNA) >t &100*rand <= PTpercent
					rnap_locs(RNA)=geneLength+10;  % mark as terminated
				end

				% PT Model 0: Rho loading with rate constant at rut sites
				if PT_Model==0 &&Ribo_locs(RNA,1) <= rut_site &&RNAP_exitTimes(rut_site,RNA) <t+dt && RNAP_exitTimes(rut_site,RNA) >t
					rut_loadT(RNA,rs_idx) = t + exprnd(1/KRutLoading);
					if Random_Model ==1
						rut_loadT(RNA,rs_idx) = t + normrnd(1/KRutLoading,sqrt(1/KRutLoading));
					end
				end

				% Load Rho onto rut site when scheduled time arrives
				% (only if ribosome hasn't passed the site and RNAP is still active)
				if t>rut_loadT(RNA,rs_idx) && rho_locs(RNA) <rut_site && Ribo_locs(RNA,1)<rut_site && rnap_locs(RNA)>rut_site && RNAP_RiboCoupling(RNA) ==0 && rnap_locs(RNA) <geneLength+1
					rho_locs(RNA) =rut_site;
				end

			end

			% If RNAP was terminated, also stop its Rho
			if rnap_locs(RNA)==geneLength+10;
				rho_locs(RNA) =geneLength+9;
			end

			%% ── Rho elongation (chases the RNAP) ───────────────────────
			if(rho_locs(RNA) >0 && rho_locs(RNA) <geneLength)
            	bases_evaluated = ceil(rutSpeed*dt*10);
				if rho_locs(RNA)+riboSpeed*5 <=geneLength
					tempExitTimes = t+cumsum(exprnd(specificDwelltimeRho(rho_locs(RNA):rho_locs(RNA)+riboSpeed*5)));
				else
					tempExitTimes = t+cumsum(exprnd(specificDwelltimeRho(rho_locs(RNA):geneLength)));
				end
				tempRho=  tempExitTimes((tempExitTimes>=t & tempExitTimes<=t+dt));
				rho_locs(RNA) = rho_locs(RNA)+size(tempRho,1);
			end
			% Rho caught the RNAP → premature termination
			if rho_locs(RNA) >=rnap_locs(RNA)
				rnap_locs(RNA)=geneLength+10;
				rho_locs(RNA) =geneLength+9;
			end
		end

		%% ── Ribosome elongation ─────────────────────────────────────────
		for RNA = 1:size(Ribo_locs,1)
            if Ribo_locs(RNA) <=geneLength &&Ribo_locs(RNA) >0
				% Sample exit times for a window of bases ahead of this ribosome
            	bases_evaluated = ceil(riboSpeed*10*dt);
			    if Ribo_locs(RNA)+bases_evaluated <=geneLength
				    tempExitTimes2 = t+cumsum(exprnd(RibospecificDwelltime1(Ribo_locs(RNA):Ribo_locs(RNA)+bases_evaluated)));
			    else
				    tempExitTimes2 = t+cumsum(exprnd(RibospecificDwelltime1(Ribo_locs(RNA):geneLength)));
			    end
				% Keep only exits within this time step [t, t+dt]
			    tempRibo_exitTimes=  tempExitTimes2((tempExitTimes2>=t & tempExitTimes2<=t+dt));

				% Coupled: ribosome moves in lockstep with RNAP
			    if RNAP_RiboCoupling(RNA)==1 && Ribo_locs(RNA) <= geneLength-RNAP_width
				    riboExitTimes(Ribo_locs(RNA):geneLength-RNAP_width,RNA) =RNAP_exitTimes(Ribo_locs(RNA)+RNAP_width:geneLength,RNA);
				    Ribo_locs(RNA) = rnap_locs(RNA)-RNAP_width;

				% Coupled but near gene end: ribosome finishes independently
			    elseif RNAP_RiboCoupling(RNA)==1 && Ribo_locs(RNA) > geneLength-Ribo_width && Ribo_locs(RNA) < geneLength+1
				    riboExitTimes(Ribo_locs(RNA):geneLength,RNA)= t+cumsum(exprnd(RibospecificDwelltime1(Ribo_locs(RNA):geneLength)));
				    Ribo_locs(RNA) = geneLength+1;

				% RNAP prematurely terminated: ribosome finishes remaining transcript
			    elseif rnap_locs(RNA) ==geneLength+10
				    riboExitTimes(Ribo_locs(RNA):sum(RNAP_exitTimes(:,RNA)>0),RNA)= t+cumsum(exprnd(RibospecificDwelltime1(Ribo_locs(RNA):sum(RNAP_exitTimes(:,RNA)>0))));
				    idx =length((RNAP_exitTimes(RNAP_exitTimes(:,RNA)>0)))+1;
				    riboExitTimes(idx:geneLength,RNA)= zeros(geneLength-idx+1,1);
				    Ribo_locs(RNA) = geneLength+10;

			    else
					% Free ribosome: check for collision with RNAP ahead
				    overlap=(Ribo_locs(RNA)+size(tempRibo_exitTimes,1)-1)-rnap_locs(RNA)+RNAP_width;
				    if rnap_locs(RNA)==geneLength+1
					    overlap =0; % RNAP finished, no collision possible
				    end
				    if overlap >0
						% Collision → couple ribosome to RNAP
					    if(rnap_locs(RNA)<=geneLength && boolRNAPRiboCoupling ==1)
						    RNAP_RiboCoupling(RNA)=1;
					    end
					    riboExitTimes(Ribo_locs(RNA):(Ribo_locs(RNA)+size(tempRibo_exitTimes,1)-1-overlap),RNA)=tempRibo_exitTimes(1:size(tempRibo_exitTimes,1)-overlap);
					    Ribo_locs(RNA) = rnap_locs(RNA)-RNAP_width;
				    else
						% No collision: advance freely
					    riboExitTimes(Ribo_locs(RNA):(Ribo_locs(RNA)+size(tempRibo_exitTimes,1)-1),RNA)=tempRibo_exitTimes;
					    Ribo_locs(RNA) = Ribo_locs(RNA)+size(tempRibo_exitTimes,1);
				    end
                end


        	end
		end
	end

	%% ════════════════════════════════════════════════════════════════════
	%  COMPUTE NETseq SIGNAL
	%  For each time t, find each RNAP's position (= how many positions it
	%  has exited by time t) and build a histogram across positions.
	%% ════════════════════════════════════════════════════════════════════
	for t= 1:simtime
		% For each RNAP: count positions with exit_time <= t and > 0 → current position
		tempNETseq = sum(RNAP_exitTimes(:,:)<=t & RNAP_exitTimes(:,:)>0,1);
		% Keep only active RNAPs (on gene, not yet finished)
		tempNETseq =tempNETseq(tempNETseq>0 &tempNETseq<geneLength & max(RNAP_exitTimes(:,:),[],1)>t);
		% Histogram of RNAP positions → NETseq density at this time
		tempNETseq = histcounts(tempNETseq,'BinMethod','integers','BinLimits',[1,geneLength]);
		NETseq(:,t)= tempNETseq;
	end


	output =struct;
	output.parameters = parameters;
	output.NETseq = NETseq;

end

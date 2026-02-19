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
%
% ALGORITHM OVERVIEW
% ==================
% This is a 1-D stochastic TASEP (Totally Asymmetric Simple Exclusion Process).
% Particles (RNAPs, ribosomes, Rho) move rightward along a 1-D lattice (the gene)
% with steric exclusion: no two particles can overlap.
%
% The core movement algorithm works like this for every particle at each dt:
%   1. Sample a "window" of exponential random dwell times for the next few bases
%   2. Compute cumulative exit times: t + cumsum(exprnd(dwelltime_per_base))
%   3. Keep only exits that fall within the current time step [t, t+dt]
%   4. The number of such exits = how many bases the particle advances this step
%   5. Check for collisions with the particle ahead; truncate advance if needed
%
% SENTINEL VALUES
% ===============
% rnap_locs values have special meanings:
%   1..geneLength        : RNAP is actively transcribing at that position
%   geneLength+1         : RNAP has finished transcribing (ran off the end)
%   geneLength+10        : RNAP was prematurely terminated by Rho
% rho_locs values:
%   0                    : no Rho loaded for this RNAP
%   1..geneLength        : Rho is chasing the RNAP at that position
%   geneLength+9         : Rho is done (RNAP was terminated)
%
% KEY DATA STRUCTURES
% ===================
% RNAP_exitTimes(pos, rnap_idx) : the time at which RNAP #rnap_idx exited position pos
%   - Rows = positions (1..geneLength), Columns = RNAP index
%   - Value > 0 means the RNAP has passed that position; 0 means not yet reached
%   - This matrix is the basis for computing the NETseq signal at the end
%
% rnap_locs(rnap_idx)  : current position of RNAP #rnap_idx
% Ribo_locs(rnap_idx)  : current position of the ribosome on RNAP #rnap_idx's mRNA
% rho_locs(rnap_idx)   : current position of Rho factor chasing RNAP #rnap_idx
%
% RNAPs are ordered by loading time: rnap_locs(1) is the first (most downstream)
% RNAP, and rnap_locs(end) is the most recently loaded (most upstream).
% For collision checks, rnap(i) looks at rnap(i-1) which is the one *ahead* of it.
%
%--------------------------------------------------------------------------------------------------

	%% ── Default parameters ──────────────────────────────────────────────
	parameters =struct;
	parameters.RNAPSpeed =19;          % RNAP elongation speed (nt/s)
	parameters.ribospeed= 19;          % ribosome elongation speed (nt/s)
	parameters.kLoading =1/20;         % RNAP initiation rate (1/s); mean wait = 20s between loadings
	parameters.kRiboLoading = 0;       % ribosome loading rate (1/s); 0 = disabled by default
	parameters.KRutLoading =0.13;      % Rho/rut loading parameter (used as PT probability in Model 1/2)
	parameters.simtime=2000;           % total simulation time (s)
	parameters.glutime=1600;           % end of active transcription (s) — after this, no new RNAPs load
	parameters.geneLength=3075;        % DNA length (bp), will be overridden by dwell profile length below
	parameters.RNAP_dwellTimeProfile = ones(parameters.geneLength,1);  % flat profile = uniform speed

	% Override defaults with any user-supplied parameters.
	% This loop copies every field from input_parameters into the local parameters struct,
	% so the caller only needs to specify the parameters they want to change.
	if nargin >=1
		parameterlabels= fieldnames(input_parameters);
		for i = 1:numel(parameterlabels)
				parameters.(parameterlabels{i})= input_parameters.(parameterlabels{i});
		end
	end
	parameters.rutSpeed=5*parameters.ribospeed;  % Rho moves 5x faster than ribosome
	% The actual gene length is determined by the dwell profile, not the default geneLength.
	% This lets the caller pass a profile of any length and the simulation adapts.
	parameters.geneLength= length(parameters.RNAP_dwellTimeProfile);

	%% ── Unpack parameters into local variables ──────────────────────────
	RNAPSpeed=parameters.RNAPSpeed;
	riboSpeed = parameters.ribospeed;
	geneLength=parameters.geneLength;
	RNAP_width=35;    % footprint of RNAP on DNA (bp) — two RNAPs cannot be closer than this
	dx=1;             % spatial resolution (bp per lattice site)
	dt=0.1;           % time step (s) — all movements are resolved within [t, t+dt] windows
	simtime=parameters.simtime;
	glutime=parameters.glutime;
	kLoading =parameters.kLoading;
	kRiboLoading = parameters.kRiboLoading;
	boolRNAPRiboCoupling=1;  % flag: if 1, ribosome-RNAP coupling is enabled on collision

	%% ── Per-RNAP state arrays ───────────────────────────────────────────
	% These arrays all grow dynamically as new RNAPs load onto the gene.
	% Column index = RNAP number (in order of loading).
	riboExitTimes =zeros(geneLength/dx,1);   % ribosome exit times at each position, per RNAP
	Ribo_locs=[];            % current position of each ribosome (one per RNAP's mRNA)
	RNAP_RiboCoupling =[0];  % coupling state per RNAP: 0=free, 1=ribosome is pushing RNAP
	Ribo_width =30;          % ribosome footprint on mRNA (bp)
	rho_width =30;           % Rho factor footprint on mRNA (bp)

	%% ── Rho / premature termination setup ───────────────────────────────
	% rut_sites: positions on the gene where Rho can potentially load.
	% Scaled proportionally so that position 500 on a 3075-bp gene maps correctly
	% to other gene lengths.
	rut_sites =[round(500*geneLength/3075)];
	rutSpeed=parameters.rutSpeed;
	% minRholoadRNA: minimum length of exposed (ribosome-unprotected) nascent RNA
	% required before Rho can load. The 80 includes the Rho footprint (30),
	% so the actual free RNA threshold is 80 - 30 = 50 nt.
	minRholoadRNA = 80-rho_width;
	rho_locs =[];            % current Rho position per RNAP (0 = no Rho loaded)
	rut_loadT=[];            % scheduled Rho loading time per RNAP per rut site
	rut_site=[1500];         % (legacy variable, used in rut_loadT initialization)
	specificDwelltimeRho=dx/rutSpeed* ones(geneLength/dx,1);  % uniform Rho dwell time per base
	tempRho=0;
	r_loc_time=zeros(2,length(rut_sites),2);
	PTpercent =0;
	% PT_Model selects which premature termination mechanism is used:
	%   0 = Rho loading at rut sites with exponential waiting time (rate-based)
	%   1 = fixed percentage chance of termination when RNAP passes a rut site
	%   2 = (DEFAULT) Rho loading probability proportional to exposed nascent RNA length
	%       This is the most biologically realistic model: longer unprotected RNA
	%       behind the RNAP = higher chance of Rho loading = more premature termination.
	PT_Model=2;
	% For Models 1 and 2, the KRutLoading parameter is repurposed as PTpercent
	% (the probability/scaling factor), and the rate-based KRutLoading is set to 0.
	if PT_Model==1 ||PT_Model==2
		KRutLoading=0;
		PTpercent =parameters.KRutLoading;
	end

	%% ── Dwell-time profiles and initial scheduling ──────────────────────
	tempExitTimes =[];
	avgDwelltime1 = dx/RNAPSpeed;       % mean RNAP dwell time per nucleotide (s/nt)
	riboavgDwelltime = dx/riboSpeed;     % mean ribosome dwell time per nucleotide (s/nt)
	% Schedule the first RNAP loading time using exponential distribution.
	% exprnd(1/kLoading) gives a random waiting time with mean = 1/kLoading seconds.
	loadt= exprnd(1/kLoading);
	rnap_locs=[];                        % current position of each RNAP on the gene
	Riboloadt= loadt + exprnd(1/kRiboLoading); % first ribosome loading time (after first RNAP loads)
	% specificDwelltime1: position-dependent RNAP dwell time array.
	% Multiply the base dwell time by the profile to make some positions slower/faster.
	% Positions with profile > 1 are slower (longer dwell), profile < 1 are faster.
	specificDwelltime1 = avgDwelltime1 .* parameters.RNAP_dwellTimeProfile;
	% RibospecificDwelltime1: ribosome dwell time is uniform (same at every position).
	RibospecificDwelltime1=riboavgDwelltime*ones(geneLength/dx,1);
	RNAP_exitTimes = zeros(geneLength/dx,1);


	%% ════════════════════════════════════════════════════════════════════
	%  MAIN SIMULATION LOOP
	%
	%  Time advances in steps of dt=0.1s from t=0 to t=simtime=2000s.
	%  The simulation has two phases:
	%    Phase 1 — Active transcription (0 ≤ t < glutime=1600s):
	%      New RNAPs load at the promoter AND existing RNAPs/ribosomes/Rho elongate.
	%    Phase 2 — Runoff (glutime ≤ t ≤ simtime):
	%      No new RNAPs load, but all existing particles continue to move.
	%      This lets us observe how the existing RNAP population drains off the gene.
	%
	%  Within each time step, four things happen in order:
	%    1. RNAP loading (Phase 1 only)
	%    2. RNAP elongation + collision detection
	%    3. Rho factor loading + elongation + premature termination check
	%    4. Ribosome elongation + RNAP-ribosome coupling
	%% ════════════════════════════════════════════════════════════════════
	for t = 0:dt:simtime

		%% ── 1. RNAP LOADING (active phase only, t < glutime) ────────────
		%
		% Loading uses a Poisson-process model: we pre-schedule the next loading
		% time (loadt) by drawing from Exp(1/kLoading). When the clock reaches
		% loadt, we attempt to load. Two outcomes:
		%
		% Case A: Promoter is BLOCKED — the most recently loaded RNAP (the last
		%   element of rnap_locs) hasn't moved far enough from position 1.
		%   Specifically, if (last_RNAP_pos - RNAP_width) <= 0, there isn't room
		%   for a new 35-bp RNAP at position 1. In this case we skip loading and
		%   reschedule loadt to a new random time in the future.
		%
		% Case B: Promoter is CLEAR — either no RNAPs exist yet, or the last one
		%   has moved at least RNAP_width bases from the promoter. We place a new
		%   RNAP at position 1 and initialize all its associated state arrays.

		% Case A: loading time has arrived, but promoter is blocked → reschedule
		if(loadt <=t & t <glutime & ~isempty(rnap_locs) & rnap_locs(length(rnap_locs))-RNAP_width <=0 )
			loadt = t + exprnd(1/kLoading);
		end

		% Case B: loading time has arrived and promoter is clear → load new RNAP
		if(loadt <=t & t <glutime &(isempty(rnap_locs) ||rnap_locs(length(rnap_locs))-RNAP_width >=0 ) )
			rnap_locs(length(rnap_locs)+1) = 1;  % new RNAP starts at position 1
			RNAP_exitTimes(:,length(rnap_locs)) = zeros(geneLength,1);  % no positions exited yet
			RNAP_RiboCoupling(length(RNAP_RiboCoupling)+1) = 0;  % not coupled to ribosome yet
			loadt = t + exprnd(1/kLoading);  % schedule next RNAP loading attempt
			Riboloadt(length(rnap_locs)) = t + exprnd(1/kRiboLoading);  % schedule ribosome for this RNAP
			Ribo_locs(length(rnap_locs),1)=0;  % ribosome not loaded yet (position 0)
			rho_locs(length(rnap_locs),1)=0;   % no Rho factor loaded yet
			riboExitTimes(:,length(rnap_locs),1)=zeros(geneLength,1);
			% Initialize rut loading times for this RNAP to simtime+1 (= never)
			% so the Rho loading condition won't trigger until explicitly scheduled.
			for rs_idx =1:size(rut_sites,2)
				rho_locs(length(rnap_locs)) =0;
				rut_loadT(length(rnap_locs),1:length(rut_site)) = simtime +1;
			end
		end

		%% ── 2. RNAP ELONGATION (runs in BOTH phases) ────────────────────
		%
		% Loop over ALL RNAPs (from first-loaded to last-loaded).
		% RNAPs are ordered so that rnap(1) is furthest downstream and rnap(end)
		% is closest to the promoter. For collision detection, rnap(i) checks
		% against rnap(i-1) which is the one directly ahead of it.
		%
		% MOVEMENT ALGORITHM (same for RNAP, ribosome, and Rho):
		%   1. Look ahead by bases_evaluated = ceil(Speed * 10 * dt) bases.
		%      The factor of 10 is a safety margin to ensure we sample enough
		%      bases even in the fastest-possible scenario within one dt.
		%   2. Draw independent Exp(specificDwelltime) for each base in the window.
		%   3. cumsum() gives the cumulative time to reach each successive base.
		%   4. Add the current time t to get absolute exit times.
		%   5. Filter to keep only exits within [t, t+dt]. These are the bases
		%      the RNAP actually crosses during this time step.
		%   6. The number of kept exits = how many bases the RNAP advances.
		%
		for rnap = 1:length(rnap_locs)
			currentRNAPloc =rnap_locs(rnap);

			% Only move this RNAP if it's still on the gene (not finished or terminated)
			if rnap_locs(rnap) <=geneLength

				% Step 1-4: Sample exit times for a window of bases ahead
				bases_evaluated =ceil(RNAPSpeed*10*dt);
				if rnap_locs(rnap)+bases_evaluated <=geneLength
					% Normal case: enough gene left for the full look-ahead window
					tempExitTimes = t+cumsum(exprnd(specificDwelltime1(currentRNAPloc:currentRNAPloc+bases_evaluated)));
				else
					% Near gene end: window extends only to the last base
					tempExitTimes = t+cumsum(exprnd(specificDwelltime1(rnap_locs(rnap):geneLength)));
				end

				% Step 5: Keep only exits within this time step [t, t+dt].
				% tempRNAP_exitTimes is a column vector; its length = number of bases advanced.
				tempRNAP_exitTimes=  tempExitTimes((tempExitTimes>=t & tempExitTimes<=t+dt));

				%% ── COLLISION DETECTION ──────────────────────────────────
				% If this isn't the first RNAP (rnap > 1), we need to check whether
				% advancing would cause it to overlap with the RNAP ahead.
				%
				% Overlap formula:
				%   overlap = (new_rear_of_current_RNAP) - (pos_of_RNAP_ahead) + RNAP_width
				%           = (currentPos + advance - 1) - PrevRNAPloc + 35
				%   If overlap > 0, the RNAPs would collide, so we truncate the advance.
				%   If overlap <= 0, there's enough room; proceed normally.
				%
				if rnap>1
					PrevRNAPloc =rnap_locs(rnap-1);

					% SPECIAL CASE: The RNAP directly ahead was prematurely terminated
					% (sentinel value geneLength+10). We need to search backwards through
					% the RNAP list to find the nearest *active* RNAP ahead.
					% Walk backwards (j=1,2,...) until we find one that isn't terminated.
					if PrevRNAPloc == geneLength + 10
						j=1;
						while j <= size(rnap_locs(1:rnap-1)) & rnap_locs(rnap-j) == geneLength + 10 &rnap-j >1;
							j =j+1;
						end
						% If we walked all the way back and every RNAP ahead is terminated,
						% treat it as if there's no obstacle (set PrevRNAPloc past gene end).
						if j == rnap || rnap-j <1
							PrevRNAPloc =geneLength+1;
						else
							PrevRNAPloc = rnap_locs(rnap-j);
						end
					end

					% Compute overlap: how far the current RNAP would intrude into the
					% space occupied by the RNAP ahead (accounting for RNAP_width footprint).
					% size(tempRNAP_exitTimes,1) = number of bases the RNAP wants to advance.
					overlap = (rnap_locs(rnap)+size(tempRNAP_exitTimes,1)-1)-PrevRNAPloc +RNAP_width;

					% If the RNAP ahead has already finished transcribing (past gene end),
					% it's no longer physically on the DNA, so no collision is possible.
					if PrevRNAPloc >= geneLength
						overlap =0;
                    end


					if overlap <=0
						% NO COLLISION: record all exit times and advance fully.
						% Write the exit times into the RNAP_exitTimes matrix at the
						% positions this RNAP just traversed.
						RNAP_exitTimes(rnap_locs(rnap):(rnap_locs(rnap)+size(tempRNAP_exitTimes,1)-1),rnap)=tempRNAP_exitTimes;
						rnap_locs(rnap) = rnap_locs(rnap)+size(tempRNAP_exitTimes,1);
					else
						% COLLISION: only advance up to the point of contact.
						% Subtract the overlap from the advance distance, so the RNAP
						% stops just behind the one ahead of it.
						RNAP_exitTimes(rnap_locs(rnap):(rnap_locs(rnap)+size(tempRNAP_exitTimes,1)-1-overlap),rnap)=tempRNAP_exitTimes(1:size(tempRNAP_exitTimes,1)-overlap);
						rnap_locs(rnap) = rnap_locs(rnap)+size(tempRNAP_exitTimes,1)-overlap;
					end
                else
					% FIRST RNAP (rnap==1): nothing ahead, so always advance freely
					% with no collision check needed.
					tempRNAP_exitTimes=  tempExitTimes((tempExitTimes>=t & tempExitTimes<=t+dt));
					RNAP_exitTimes(rnap_locs(rnap):(rnap_locs(rnap)+size(tempRNAP_exitTimes,1)-1),rnap)=tempRNAP_exitTimes;
					rnap_locs(rnap) = rnap_locs(rnap)+size(tempRNAP_exitTimes,1);
				end
			end

			%% ── RIBOSOME LOADING onto this RNAP's mRNA ──────────────────
			% A ribosome loads onto the nascent mRNA once:
			%   (a) Its scheduled loading time (Riboloadt) has arrived, AND
			%   (b) The RNAP has moved far enough (>= Ribo_width = 30 bp) so there's
			%       enough exposed mRNA for the ribosome to bind.
			% After loading, Riboloadt is set to simtime (effectively "never again")
			% because this model allows only one ribosome per mRNA.
			% The ribosome starts at position 1 on the mRNA.
			%
			% The if/elseif/else below handles MATLAB array initialization edge cases:
			%   - If Ribo_locs is empty, initialize it first
			%   - If Ribo_locs has fewer rows than the current RNAP index, expand it
			%   - Otherwise, just set position and reset exit times
			if(Riboloadt(rnap) <=t & rnap_locs(rnap) >=Ribo_width) 
				Riboloadt(rnap) = simtime; % Reset loading time to "never again"
				if size(Ribo_locs)==0 % If Ribo_locs is empty, initialize it first
					Ribo_locs(1)=0;
					riboExitTimes(1,1) =0;
					Ribo_locs(rnap) = 1;
					riboExitTimes(1,rnap)= t;
				elseif size(Ribo_locs,1) <rnap % If Ribo_locs has fewer rows than the current RNAP index, expand it
					Ribo_locs(rnap)=1;
					riboExitTimes(1,rnap) =t;
				else % Otherwise, just set position and reset exit times
					Ribo_locs(rnap)=1;
					riboExitTimes(:,rnap) = zeros(geneLength,1);
				end
            end


		end

		%% ── 3. RHO FACTOR / PREMATURE TERMINATION ───────────────────────
		%
		% Loop over every RNAP to handle Rho-dependent premature termination.
		% This section does three things per RNAP:
		%   (a) Decide whether to load a new Rho factor (depends on PT_Model)
		%   (b) Move any existing Rho factor toward the RNAP
		%   (c) Check if Rho has caught the RNAP → terminate it
		%
		for RNA = 1:length(rnap_locs)

			% ── (a) Rho loading decision ─────────────────────────────────
			%
			% PT_Model==2 (default): Probability-based Rho loading that depends on
			% how much naked (ribosome-unprotected) RNA trails behind the RNAP.
			%
			% The exposed RNA length is:
			%   PTRNAsize = RNAP_position - RNAP_footprint - Rho_footprint - Ribosome_position
			%
			% Biological intuition: if the ribosome is close behind the RNAP,
			% little RNA is exposed and Rho can't bind. If the ribosome falls behind
			% (or never loaded), more RNA is exposed and Rho loading becomes more likely.
			%
			% The loading probability per time step scales as:
			%   P = (PTpercent * PTRNAsize / geneLength) * dt
			% So longer exposed RNA and larger PTpercent both increase termination risk.
			%
			% If Rho loads, it's placed at a random position within the exposed region.
			% The "if temp_rho_loading_loc > rho_locs" check ensures Rho only advances
			% forward (never jumps backward).
			if PT_Model==2 && rnap_locs(RNA)<geneLength
				PTRNAsize = rnap_locs(RNA)- RNAP_width-rho_width;
				PTRNAsize= PTRNAsize - Ribo_locs(RNA,1);
				if PTRNAsize>minRholoadRNA && 100*dt*rand <= PTpercent*PTRNAsize/geneLength
					temp_rho_loading_loc = rnap_locs(RNA)- RNAP_width- floor(rand*PTRNAsize);
					if temp_rho_loading_loc>rho_locs(RNA,1)
						rho_locs(RNA,1)=temp_rho_loading_loc;
					end
				end
			end

			% Loop over each rut site for the alternative PT models (0 and 1).
			% These only trigger when the RNAP passes through a rut site during
			% this time step (exit time at rut_site falls within [t, t+dt]).
			for rs_idx =1:size(rut_sites,1)
				rut_site= rut_sites(rs_idx);

				% PT_Model==1: Simple percentage-based termination.
				% When the RNAP crosses a rut site, roll a random number.
				% If 100*rand <= PTpercent, terminate immediately (set to sentinel).
				if  PT_Model==1 && RNAP_exitTimes(rut_site,RNA) <t+dt && RNAP_exitTimes(rut_site,RNA) >t &100*rand <= PTpercent
					rnap_locs(RNA)=geneLength+10;
				end

				% PT_Model==0: Rate-based Rho loading at rut sites.
				% When the RNAP crosses the rut site AND the ribosome hasn't passed it yet,
				% schedule a future Rho loading time using Exp(1/KRutLoading).
				% (Random_Model==1 variant uses normal distribution instead.)
				if PT_Model==0 &&Ribo_locs(RNA,1) <= rut_site &&RNAP_exitTimes(rut_site,RNA) <t+dt && RNAP_exitTimes(rut_site,RNA) >t
					rut_loadT(RNA,rs_idx) = t + exprnd(1/KRutLoading);
					if Random_Model ==1
						rut_loadT(RNA,rs_idx) = t + normrnd(1/KRutLoading,sqrt(1/KRutLoading));
					end
				end

				% Execute the scheduled Rho loading (for PT_Model==0).
				% All six conditions must be met simultaneously:
				%   1. t > rut_loadT        : the scheduled loading time has arrived
				%   2. rho_locs < rut_site   : Rho hasn't already been placed past this site
				%   3. Ribo_locs < rut_site  : ribosome hasn't protected this site yet
				%   4. rnap_locs > rut_site  : RNAP has already passed the site
				%   5. coupling == 0         : RNAP isn't coupled to ribosome (if coupled,
				%                              ribosome is protecting the RNA)
				%   6. rnap_locs < geneLength+1 : RNAP hasn't finished transcribing
				if t>rut_loadT(RNA,rs_idx) && rho_locs(RNA) <rut_site && Ribo_locs(RNA,1)<rut_site && rnap_locs(RNA)>rut_site && RNAP_RiboCoupling(RNA) ==0 && rnap_locs(RNA) <geneLength+1
					rho_locs(RNA) =rut_site;
				end

			end

			% If this RNAP was terminated (by any PT model above), set its Rho to
			% the "done" sentinel (geneLength+9) so Rho movement code is skipped.
			if rnap_locs(RNA)==geneLength+10;
				rho_locs(RNA) =geneLength+9;
			end

			% ── (b) Rho elongation (chases the RNAP) ────────────────────
			% Same movement algorithm as RNAP (window sampling + cumsum + filter),
			% but using the Rho-specific dwell time (5x faster than ribosome).
			% Rho only moves if it's actively on the gene (0 < pos < geneLength).
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

			% ── (c) Rho catch check ─────────────────────────────────────
			% If Rho's position has reached or passed the RNAP's position,
			% Rho has "caught" the RNAP → premature termination.
			% Both are set to their respective sentinel values.
			if rho_locs(RNA) >=rnap_locs(RNA)
				rnap_locs(RNA)=geneLength+10;
				rho_locs(RNA) =geneLength+9;
			end
		end

		%% ── 4. RIBOSOME ELONGATION ──────────────────────────────────────
		%
		% Loop over every ribosome that has been loaded (Ribo_locs > 0).
		% Movement uses the same window-sampling algorithm as RNAP.
		%
		% The ribosome has four possible states, handled by the if/elseif chain:
		%
		%   State 1 — COUPLED, mid-gene:
		%     Ribosome is physically pushing the RNAP (they collided earlier).
		%     Instead of independent movement, copy the RNAP's exit times
		%     (shifted by RNAP_width) as the ribosome's exit times.
		%     The ribosome position tracks (RNAP_pos - RNAP_width).
		%
		%   State 2 — COUPLED, near gene end:
		%     Ribosome is within Ribo_width (30bp) of the gene end.
		%     The RNAP may have already run off, so the ribosome finishes
		%     independently with its own stochastic exit times.
		%
		%   State 3 — RNAP TERMINATED:
		%     The RNAP was prematurely terminated by Rho (sentinel geneLength+10).
		%     The ribosome translates the remaining transcript up to the last
		%     position the RNAP reached (= count of nonzero RNAP exit times),
		%     then zeros out exit times beyond that (no more mRNA to translate).
		%     Ribosome is then marked as terminated (geneLength+10).
		%
		%   State 4 — FREE (not coupled):
		%     Normal independent movement. After advancing, check for collision
		%     with the RNAP ahead:
		%       - If overlap > 0: collision detected → set RNAP_RiboCoupling=1,
		%         truncate ribosome advance, snap ribosome to (RNAP_pos - RNAP_width).
		%       - If overlap <= 0: no collision, advance normally.
		%     Exception: if RNAP has finished (pos == geneLength+1), overlap is
		%     forced to 0 because the RNAP is no longer physically present.
		%
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

				% --- State 1: Coupled, mid-gene ---
				% Copy RNAP exit times (offset by RNAP_width) as ribosome exit times.
				% This makes the ribosome and RNAP move as a single unit.
			    if RNAP_RiboCoupling(RNA)==1 && Ribo_locs(RNA) <= geneLength-RNAP_width
				    riboExitTimes(Ribo_locs(RNA):geneLength-RNAP_width,RNA) =RNAP_exitTimes(Ribo_locs(RNA)+RNAP_width:geneLength,RNA);
				    Ribo_locs(RNA) = rnap_locs(RNA)-RNAP_width;

				% --- State 2: Coupled, near gene end ---
				% Too close to gene end for the offset copy to work.
				% Ribosome finishes the remaining bases independently.
			    elseif RNAP_RiboCoupling(RNA)==1 && Ribo_locs(RNA) > geneLength-Ribo_width && Ribo_locs(RNA) < geneLength+1
				    riboExitTimes(Ribo_locs(RNA):geneLength,RNA)= t+cumsum(exprnd(RibospecificDwelltime1(Ribo_locs(RNA):geneLength)));
				    Ribo_locs(RNA) = geneLength+1;

				% --- State 3: RNAP prematurely terminated ---
				% The ribosome translates up to the last position the RNAP transcribed.
				% sum(RNAP_exitTimes(:,RNA)>0) counts how many positions have nonzero exit
				% times, which equals the last position the RNAP reached before termination.
				% Everything beyond that is zeroed out (no transcript exists there).
			    elseif rnap_locs(RNA) ==geneLength+10
				    riboExitTimes(Ribo_locs(RNA):sum(RNAP_exitTimes(:,RNA)>0),RNA)= t+cumsum(exprnd(RibospecificDwelltime1(Ribo_locs(RNA):sum(RNAP_exitTimes(:,RNA)>0))));
				    idx =length((RNAP_exitTimes(RNAP_exitTimes(:,RNA)>0)))+1;
				    riboExitTimes(idx:geneLength,RNA)= zeros(geneLength-idx+1,1);
				    Ribo_locs(RNA) = geneLength+10;

				% --- State 4: Free ribosome ---
			    else
					% Compute overlap between ribosome's new position and the RNAP ahead.
					% Same formula as RNAP collision: positive overlap = collision.
				    overlap=(Ribo_locs(RNA)+size(tempRibo_exitTimes,1)-1)-rnap_locs(RNA)+RNAP_width;
					% If RNAP has finished (ran off gene end), it's not physically there,
					% so no collision is possible.
				    if rnap_locs(RNA)==geneLength+1
					    overlap =0;
				    end
				    if overlap >0
						% COLLISION: ribosome has caught up to the RNAP.
						% Enable coupling so they move together from now on.
					    if(rnap_locs(RNA)<=geneLength && boolRNAPRiboCoupling ==1)
						    RNAP_RiboCoupling(RNA)=1;
					    end
						% Record exit times only for the bases before the collision point,
						% then snap ribosome position to just behind the RNAP.
					    riboExitTimes(Ribo_locs(RNA):(Ribo_locs(RNA)+size(tempRibo_exitTimes,1)-1-overlap),RNA)=tempRibo_exitTimes(1:size(tempRibo_exitTimes,1)-overlap);
					    Ribo_locs(RNA) = rnap_locs(RNA)-RNAP_width;
				    else
						% NO COLLISION: ribosome advances freely.
					    riboExitTimes(Ribo_locs(RNA):(Ribo_locs(RNA)+size(tempRibo_exitTimes,1)-1),RNA)=tempRibo_exitTimes;
					    Ribo_locs(RNA) = Ribo_locs(RNA)+size(tempRibo_exitTimes,1);
				    end
                end


        	end
		end
	end

	%% ════════════════════════════════════════════════════════════════════
	%  COMPUTE NETseq SIGNAL
	%
	%  After the simulation, RNAP_exitTimes contains the time each RNAP
	%  exited each position. We now convert this into a NETseq density
	%  matrix: NETseq(position, time) = number of RNAPs at that position
	%  at that time.
	%
	%  Algorithm for each integer time t (1s, 2s, ..., simtime):
	%
	%  Step 1: For each RNAP, count how many positions have exit_time > 0
	%    and exit_time <= t. This count IS the RNAP's current position at
	%    time t (because exit times are cumulative from position 1 onward).
	%    sum(...<=t & ...>0, 1) sums along rows → gives a 1 x N_rnaps vector.
	%
	%  Step 2: Filter out RNAPs that:
	%    - Haven't loaded yet (position = 0)
	%    - Have already run off the gene (position >= geneLength)
	%    - Have finished transcribing (max exit time <= t, meaning the RNAP
	%      has exited all positions by time t)
	%
	%  Step 3: Histogram the remaining positions into integer bins [1..geneLength].
	%    histcounts with 'BinMethod','integers' creates one bin per integer position.
	%    The result is the NETseq density profile at time t.
	%
	%  The output NETseq matrix is (geneLength x simtime).
	%% ════════════════════════════════════════════════════════════════════
	for t= 1:simtime
		% Step 1: For each RNAP, find its current position at time t
		tempNETseq = sum(RNAP_exitTimes(:,:)<=t & RNAP_exitTimes(:,:)>0,1);
		% Step 2: Keep only active RNAPs still on the gene
		tempNETseq =tempNETseq(tempNETseq>0 &tempNETseq<geneLength & max(RNAP_exitTimes(:,:),[],1)>t);
		% Step 3: Build histogram of RNAP positions → NETseq density at time t
		tempNETseq = histcounts(tempNETseq,'BinMethod','integers','BinLimits',[1,geneLength]);
		NETseq(:,t)= tempNETseq;
	end


	output =struct;
	output.parameters = parameters;
	output.NETseq = NETseq;

end

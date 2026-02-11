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

	parameters =struct;
	parameters.RNAPSpeed =19;parameters.ribospeed= 19;
	parameters.kLoading =1/20;parameters.kRiboLoading = 0;parameters.KRutLoading =0.13;
	parameters.simtime=2000;parameters.glutime=1600; 
	parameters.geneLength=3075;%DNA length 
	parameters.RNAP_dwellTimeProfile = ones(parameters.geneLength,1);
	if nargin >=1 %number of arguments in
		parameterlabels= fieldnames(input_parameters);
		for i = 1:numel(parameterlabels)
				parameters.(parameterlabels{i})= input_parameters.(parameterlabels{i});
		end
	end
	parameters.rutSpeed=5*parameters.ribospeed;
	parameters.geneLength= length(parameters.RNAP_dwellTimeProfile);


	RNAPSpeed=parameters.RNAPSpeed;riboSpeed = parameters.ribospeed;
	geneLength=parameters.geneLength;   
	RNAP_width=35; %in bp
	dx=1;dt=0.1;
	simtime=parameters.simtime;glutime=parameters.glutime; %glu= glucose, sim = simulation 
	kLoading =parameters.kLoading;kRiboLoading = parameters.kRiboLoading;
	boolRNAPRiboCoupling=1; 

	% Input parameters for ribosome initiation and elongation
	riboExitTimes =zeros(geneLength/dx,1); % lets make it a 3d array such as [bp,RNAP#,Ribo#] %ceil(glutime*kLoading)
	Ribo_locs=[]; % 2d array [RNAP#, Ribo#]
	RNAP_RiboCoupling =[0];% 1 for its coupled 0 for not
	Ribo_width =30;
	rho_width =30;


	%Distribution of mRNA numbers (FISH)
	

	%rut sites
	%% we dont need to track the exittimes we just need to keep the rho positions
	rut_sites =[round(500*geneLength/3075)];
	rutSpeed=parameters.rutSpeed;
	minRholoadRNA = 80-rho_width;
	rho_locs =[];
	rut_loadT=[];
	rut_site=[1500];
	specificDwelltimeRho=dx/rutSpeed* ones(geneLength/dx,1);
	tempRho=0;
	r_loc_time=zeros(2,length(rut_sites),2);
	PTpercent =0;
	PT_Model=2;
	if PT_Model==1 ||PT_Model==2
		KRutLoading=0;
		PTpercent =parameters.KRutLoading;
	end

	tempExitTimes =[]; %rename to temptimes or something
	avgDwelltime1 = dx/RNAPSpeed; %sec per nucleotide
	riboavgDwelltime = dx/riboSpeed;
	loadt= exprnd(1/kLoading); %*rand;
	rnap_locs=[];
	Riboloadt= loadt + exprnd(1/kRiboLoading); % 1d array of size sz(RNAP_locs)
	specificDwelltime1 = avgDwelltime1 .* parameters.RNAP_dwellTimeProfile;
	RibospecificDwelltime1=riboavgDwelltime*ones(geneLength/dx,1);
	RNAP_exitTimes = zeros(geneLength/dx,1);


	for t = 0:dt:simtime
		%%RNAP loading
		%%lets test some random bs
		
		if(loadt <=t & t <glutime & ~isempty(rnap_locs) & rnap_locs(length(rnap_locs))-RNAP_width <=0 )  %% make sure no rnap on the site at loading time.
			loadt = t + exprnd(1/kLoading);% calculate load time for next RNAP
		end
		
		if(loadt <=t & t <glutime &(isempty(rnap_locs) ||rnap_locs(length(rnap_locs))-RNAP_width >=0 ) ) %% make sure no rnap on the site at loading time.
			rnap_locs(length(rnap_locs)+1) = 1;
			RNAP_exitTimes(:,length(rnap_locs)) = zeros(geneLength,1);
			RNAP_RiboCoupling(length(RNAP_RiboCoupling)+1) = 0;
			loadt = t + exprnd(1/kLoading);% calculate load time for next RNAP
			Riboloadt(length(rnap_locs)) = t + exprnd(1/kRiboLoading); %calculate first ribo load time on that RNAP
			Ribo_locs(length(rnap_locs),1)=0;rho_locs(length(rnap_locs),1)=0;
			riboExitTimes(:,length(rnap_locs),1)=zeros(geneLength,1);
			for rs_idx =1:size(rut_sites,2)
				rho_locs(length(rnap_locs)) =0; %initialize that this can have rut sites.
				rut_loadT(length(rnap_locs),1:length(rut_site)) = simtime +1; % initialize the loadT array but we will set the correct loadt later
			end
		end

		%%RNAP loop
		for rnap = 1:length(rnap_locs) 
			currentRNAPloc =rnap_locs(rnap);
			if rnap_locs(rnap) <=geneLength
				bases_evaluated =ceil(RNAPSpeed*10*dt);
				if rnap_locs(rnap)+bases_evaluated <=geneLength % add in integer(value*dt)
					tempExitTimes = t+cumsum(exprnd(specificDwelltime1(currentRNAPloc:currentRNAPloc+bases_evaluated))); %RNAP_exitTimes(currentRNAPloc,rnap)
				else
					tempExitTimes = t+cumsum(exprnd(specificDwelltime1(rnap_locs(rnap):geneLength)));
				end

				tempRNAP_exitTimes=  tempExitTimes((tempExitTimes>=t & tempExitTimes<=t+dt));	

				if rnap>1
					PrevRNAPloc =rnap_locs(rnap-1);
					if PrevRNAPloc == geneLength + 10
						j=1;
						while j <= size(rnap_locs(1:rnap-1)) & rnap_locs(rnap-j) == geneLength + 10 &rnap-j >1;
							j =j+1;
						end
						if j == rnap || rnap-j <1 %% if there is no rnap behind thats not terminated
							PrevRNAPloc =geneLength+1;
						else
							PrevRNAPloc = rnap_locs(rnap-j);
						end
					end
					overlap = (rnap_locs(rnap)+size(tempRNAP_exitTimes,1)-1)-PrevRNAPloc +RNAP_width; %% check for collision overlapt is +ve if collision
					
					if PrevRNAPloc >= geneLength
						overlap =0;  %if the previous rnap is done transcribing it cant collide.
                    end

                    
					if overlap <=0 
						RNAP_exitTimes(rnap_locs(rnap):(rnap_locs(rnap)+size(tempRNAP_exitTimes,1)-1),rnap)=tempRNAP_exitTimes;
						rnap_locs(rnap) = rnap_locs(rnap)+size(tempRNAP_exitTimes,1);
					else
						RNAP_exitTimes(rnap_locs(rnap):(rnap_locs(rnap)+size(tempRNAP_exitTimes,1)-1-overlap),rnap)=tempRNAP_exitTimes(1:size(tempRNAP_exitTimes,1)-overlap);
						rnap_locs(rnap) = rnap_locs(rnap)+size(tempRNAP_exitTimes,1)-overlap;
					end
                else
					tempRNAP_exitTimes=  tempExitTimes((tempExitTimes>=t & tempExitTimes<=t+dt));
					RNAP_exitTimes(rnap_locs(rnap):(rnap_locs(rnap)+size(tempRNAP_exitTimes,1)-1),rnap)=tempRNAP_exitTimes;
					rnap_locs(rnap) = rnap_locs(rnap)+size(tempRNAP_exitTimes,1);
				end
			end

			%%Ribosome loading
			
			if(Riboloadt(rnap) <=t & rnap_locs(rnap) >=Ribo_width)
				Riboloadt(rnap) = simtime;
				if size(Ribo_locs)==0
					Ribo_locs(1)=0; %floor(Ribo_width/2+1);
					riboExitTimes(1,1) =0;
					Ribo_locs(rnap) = 1;
					riboExitTimes(1,rnap)= t;
				elseif size(Ribo_locs,1) <rnap
					Ribo_locs(rnap)=1;%floor(Ribo_width/2+1);
					riboExitTimes(1,rnap) =t;
				else
					Ribo_locs(rnap)=1;
					riboExitTimes(:,rnap) = zeros(geneLength,1);
				end
            end

			
		end

		%%rhofactor simulation.
		for RNA = 1:length(rnap_locs)
			%for loop for RUT site PT cases.

			%lets add in some cool mechanics for the PT percentage based on free RNA behind the RNAP.
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
			for rs_idx =1:size(rut_sites,1) %rs_idx= rut site idx
				rut_site= rut_sites(rs_idx);
				%RUT site mechanics with Percentage Premature termination.
				if  PT_Model==1 && RNAP_exitTimes(rut_site,RNA) <t+dt && RNAP_exitTimes(rut_site,RNA) >t &100*rand <= PTpercent	
					rnap_locs(RNA)=geneLength+10;
					%rho_locs(RNA,rs_idx) =geneLength+9;
				end
				
				% RUT site mechanics with kloading
				if PT_Model==0 &&Ribo_locs(RNA,1) <= rut_site &&RNAP_exitTimes(rut_site,RNA) <t+dt && RNAP_exitTimes(rut_site,RNA) >t %%rutsite loadt calculation
					
					rut_loadT(RNA,rs_idx) = t + exprnd(1/KRutLoading);
					if Random_Model ==1 
						rut_loadT(RNA,rs_idx) = t + normrnd(1/KRutLoading,sqrt(1/KRutLoading));
					end
				end

				%load the rut on rutsite when loadt happens
				%%if ribo is ahead no point in having rut load so we wont include that case.
				if t>rut_loadT(RNA,rs_idx) && rho_locs(RNA) <rut_site && Ribo_locs(RNA,1)<rut_site && rnap_locs(RNA)>rut_site && RNAP_RiboCoupling(RNA) ==0 && rnap_locs(RNA) <geneLength+1
					rho_locs(RNA) =rut_site;
				end


			end

			%check if rnap already terminated
			if rnap_locs(RNA)==geneLength+10;
				rho_locs(RNA) =geneLength+9; % no need to have the rho continue till end lets just have it terminate
				%r_loc_time(RNA,floor(t/dt))=0;
			end
			%%calculate rho movement.
			if(rho_locs(RNA) >0 && rho_locs(RNA) <geneLength)
            	bases_evaluated = ceil(rutSpeed*dt*10);
				if rho_locs(RNA)+riboSpeed*5 <=geneLength
					tempExitTimes = t+cumsum(exprnd(specificDwelltimeRho(rho_locs(RNA):rho_locs(RNA)+riboSpeed*5)));	
				else
					tempExitTimes = t+cumsum(exprnd(specificDwelltimeRho(rho_locs(RNA):geneLength)));	
				end
				tempRho=  tempExitTimes((tempExitTimes>=t & tempExitTimes<=t+dt));
				rho_locs(RNA) = rho_locs(RNA)+size(tempRho,1);
				%r_loc_time(RNA,floor(t/dt))=rho_locs(RNA);
			end
			if rho_locs(RNA) >=rnap_locs(RNA)
				rnap_locs(RNA)=geneLength+10;
				rho_locs(RNA) =geneLength+9;
			end
		end

		%%ribo simulation
		for RNA = 1:size(Ribo_locs,1)
            if Ribo_locs(RNA) <=geneLength &&Ribo_locs(RNA) >0
            	bases_evaluated = ceil(riboSpeed*10*dt);
			    if Ribo_locs(RNA)+bases_evaluated <=geneLength
				    tempExitTimes2 = t+cumsum(exprnd(RibospecificDwelltime1(Ribo_locs(RNA):Ribo_locs(RNA)+bases_evaluated)));	
			    else
				    tempExitTimes2 = t+cumsum(exprnd(RibospecificDwelltime1(Ribo_locs(RNA):geneLength)));	
			    end
			    tempRibo_exitTimes=  tempExitTimes2((tempExitTimes2>=t & tempExitTimes2<=t+dt));
			    %%add in ribosome collision
			    %%add in ribsome exit time code
			    if RNAP_RiboCoupling(RNA)==1 && Ribo_locs(RNA) <= geneLength-RNAP_width
				    %RNAP and RIbo are coupled so link their movements together
				    riboExitTimes(Ribo_locs(RNA):geneLength-RNAP_width,RNA) =RNAP_exitTimes(Ribo_locs(RNA)+RNAP_width:geneLength,RNA); % this is overkill u only really need to copy till current RNAPloc but this also works just computationally worse.
				    Ribo_locs(RNA) = rnap_locs(RNA)-RNAP_width;
			    elseif RNAP_RiboCoupling(RNA)==1 && Ribo_locs(RNA) > geneLength-Ribo_width && Ribo_locs(RNA) < geneLength+1
				    %compute the last bits of the ribo translating on RNAP
				    riboExitTimes(Ribo_locs(RNA):geneLength,RNA)= t+cumsum(exprnd(RibospecificDwelltime1(Ribo_locs(RNA):geneLength)));
				    Ribo_locs(RNA) = geneLength+1;
			    elseif rnap_locs(RNA) ==geneLength+10 % premature termination code for the ribosome movement
				    riboExitTimes(Ribo_locs(RNA):sum(RNAP_exitTimes(:,RNA)>0),RNA)= t+cumsum(exprnd(RibospecificDwelltime1(Ribo_locs(RNA):sum(RNAP_exitTimes(:,RNA)>0))));
				    idx =length((RNAP_exitTimes(RNAP_exitTimes(:,RNA)>0)))+1;
                    % could be idx = size(RNAP_exitTimes(:,RNA)>0))+1
				    riboExitTimes(idx:geneLength,RNA)= zeros(geneLength-idx+1,1);
				    Ribo_locs(RNA) = geneLength+10;
			    else
				    overlap=(Ribo_locs(RNA)+size(tempRibo_exitTimes,1)-1)-rnap_locs(RNA)+RNAP_width;
				    if rnap_locs(RNA)==geneLength+1
					    overlap =0; %if rnap is done transcribing you cant overlap with the rnap
				    end
				    if overlap >0	%if collided then lets have them coupled.
					    if(rnap_locs(RNA)<=geneLength && boolRNAPRiboCoupling ==1)
						    RNAP_RiboCoupling(RNA)=1;
					    end
					    riboExitTimes(Ribo_locs(RNA):(Ribo_locs(RNA)+size(tempRibo_exitTimes,1)-1-overlap),RNA)=tempRibo_exitTimes(1:size(tempRibo_exitTimes,1)-overlap);
					    Ribo_locs(RNA) = rnap_locs(RNA)-RNAP_width;
				    else
					    riboExitTimes(Ribo_locs(RNA):(Ribo_locs(RNA)+size(tempRibo_exitTimes,1)-1),RNA)=tempRibo_exitTimes;
					    Ribo_locs(RNA) = Ribo_locs(RNA)+size(tempRibo_exitTimes,1);
				    end
                end
                   
			    
        	end
		end
	end

	for t= 1:simtime
		tempNETseq = sum(RNAP_exitTimes(:,:)<=t & RNAP_exitTimes(:,:)>0,1);
		tempNETseq =tempNETseq(tempNETseq>0 &tempNETseq<geneLength & max(RNAP_exitTimes(:,:),[],1)>t);
		tempNETseq = histcounts(tempNETseq,'BinMethod','integers','BinLimits',[1,geneLength]);
		NETseq(:,t)= tempNETseq;
	end


	output =struct;
	output.parameters = parameters;
	output.NETseq = NETseq;

end
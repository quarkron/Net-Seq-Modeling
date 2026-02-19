%--------------------------------------------------------------------------------------------------
% Code written by: Albur Hassan
% sjkimlab at University of Illinois at Urbana Champaign.
% Original creation date: 2026/01/25
% Last edited: 2026/01/26
% Contact: ahassan4@illinois.edu
%--------------------------------------------------------------------------------------------------
% Program Description: this script works to call the sjkimlab_NETSEQ_TASEP function multiple times
% and aggregate the results for further analysis
%--------------------------------------------------------------------------------------------------
% To run: specify the gene you are interested in by editting the gene name in the gene variable and
% the update the number of loci simulated in the nloci variable. Then run this script file and it 
% should produce a graph with the output NETSEQ results of the simulation.
%--------------------------------------------------------------------------------------------------
% This script requires the following files and folder:
% sjkimlab_NETSEQ_TASEP.m
% Ecoli_gene_TE.csv
% a folder named NETSEQ_gene which multiple csv files labeled as NETSEQ_geneName.csv to use for
% the RNAP dwell times in the simulation.
%--------------------------------------------------------------------------------------------------

gene = 'insQ';
nloci =300;

genefname = fullfile(fileparts(mfilename('fullpath')), "NETSEQ_gene", "NETSEQ_"+gene+".csv");


%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 3);
% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";
% Specify column names and types
opts.VariableNames = ["Gene", "mRNALevelRPKM", "TranslationEfficiencyAU"];
opts.VariableTypes = ["string", "double", "double"];
% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
% Specify variable properties
opts = setvaropts(opts, "Gene", "WhitespaceRule", "preserve");
opts = setvaropts(opts, "Gene", "EmptyFieldRule", "auto");
% Import the data
scriptDir = fileparts(mfilename('fullpath'));
EcoligeneTE = readtable(fullfile(scriptDir, "Ecoli_gene_TE.csv"), opts);
%% Clear temporary variables
clear opts

kribo=  EcoligeneTE.TranslationEfficiencyAU(EcoligeneTE.Gene==gene)/5;
if isnan(kribo)
	kribo=0;
end

%create parameter struct
parameters= struct;
parameters.KRutLoading=0.13;
parameters.RNAP_dwellTimeProfile= readmatrix(genefname);
parameters.RNAP_dwellTimeProfile= parameters.RNAP_dwellTimeProfile./mean(parameters.RNAP_dwellTimeProfile);
parameters.kRiboLoading= kribo;
%run simulation loop
final_output = sjkimlab_NETSEQ_TASEP(parameters);
for i =1:nloci
	output = sjkimlab_NETSEQ_TASEP(parameters);
	final_output.NETseq = final_output.NETseq+ output.NETseq;
end
final_output.NETseq = final_output.NETseq/nloci;
temp_NETseq = final_output.NETseq;
NETseqSum = sum(temp_NETseq(:,200:200:1500),2);

figure; hold on;
plot(NETseqSum);

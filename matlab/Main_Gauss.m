%==========================================================================
% clc;
clear;
close all;
addpath(genpath('lib'));

%% Data init
dataname = 'ICVL';
basedir = ['../data/', dataname];
datadir = [basedir, '/filefolder/test_gauss'];
if isempty(gcp)
    parpool(4,'IdleTimeout', inf); % If your computer's memory is less than 8G, do not use more than 4 workers.
end

%%
methodname = {'None','BM4D','TDL', 'ITSReg','LLRT'};
ds_names = dir([basedir, '/filefolder/testset_gauss']);
ds_names = {ds_names(3:end).name};

for i = 1:length(ds_name)
    dataset_name = ds_names{i};
    datadir_current = fullfile(datadir, dataset_name);
    fns = dir(datadir_current+'/*.mat');
    fns = {fns.name};
    resdir = fullfile('Result', dataset_name);
    HSI_test(datadir_current, resdir, fns, methodname);
end

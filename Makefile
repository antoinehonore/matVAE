SHELL=/bin/bash

include cfg.mk 
#PYTHON=python



ifndef verbose
	verbose=0
endif

ifndef n_jobs
	n_jobs=1
endif

ifndef script
	script=main.py
endif

ifndef CARGS
	CARGS=
endif

cfg_dir=configs
res_dir=results

usage="USAGE:\nmake init indir=example\nmake script=main.py indir=example verbose=1 n_jobs=<Number of tasks per process> -j <Total number of processes>"

all_configs=$(shell ls $(cfg_dir)/$(indir))
all_targets=$(addprefix $(res_dir)/$(indir)/,$(foreach fname,$(all_configs),$(shell echo $(fname) | sed 's/.json/.pkl/g')))
all_converted=$(addprefix $(res_dir)/$(indir)/,$(foreach fname,$(all_configs),$(shell echo $(fname) | sed 's/.json/.csv/g')))

all: $(all_targets)
	@echo $^

help:
	@echo -e $(usage)

init: $(cfg_dir)/$(indir).json
	rm -f $(cfg_dir)/$(indir)/*.json
	$(PYTHON) gen_json.py -i $^
	find $(cfg_dir)/$(indir) -type f -name "*.json" | wc -w

# User specific targets
$(res_dir)/$(indir)/%.pkl: $(cfg_dir)/$(indir)/%.json
	$(SLURM) $(PYTHON) $(script) -i $^ -v $(verbose) -j $(n_jobs) $(CARGS)



# These targets will catch the cases where you specify the name of the config file as a target.
%:
	make indir=$* n_jobs=$(n_jobs) script=$(script) verbose=$(verbose) CARGS=$(CARGS) SLURM="$(SLURM)" -j $(n_jobs) 

%-init:
	make indir=$* init n_jobs=$(n_jobs) script=$(script) verbose=$(verbose)

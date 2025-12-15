IMAGE_NAME = prop_hunt

IMAGE_EXISTS := $(shell docker image inspect ${IMAGE_NAME} >/dev/null 2>&1; echo $$?;)
ifeq ($(IMAGE_EXISTS), 1)
	BUILD_TARG = docker
endif

.PHONY: docker data plots

docker:
	docker build -t ${IMAGE_NAME} --platform linux/amd64 .

data: $(BUILD_TARG)
	./run_experiments.sh

plots:
	cd scripts; \
	python plot_data.py

##########
# Docker #
##########

SHELL := /bin/bash

# variables
WORK_DIR = $(PWD)
USER_ID = $$(id -u)
GROUP_ID = $$(id -g)

DOCKER_BUILD_ARGS = \
	--build-arg USER_ID=$(USER_ID) \
	--build-arg GROUP_ID=$(GROUP_ID) \
	--build-arg GITHUB_USER_TOKEN=$(GITHUB_USER_TOKEN) \
	--build-arg GITHUB_ACCESS_TOKEN=$(GITHUB_ACCESS_TOKEN)

DOCKER_RUN_FLAGS = --rm --privileged -p ${PORT}:${PORT} --network host

DOCKER_IMAGE_NAME = popcorn2
DOCKER_CONTAINER_NAME = popcorn2_container


.PHONY: clean
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rfv
	find . | grep -E ".pytest_cache" | xargs rm -rfv
	find . | grep -E "nul" | xargs rm -rfv

.PHONY: docker_build_tpu
docker_build_tpu:
	sudo docker build -t $(DOCKER_IMAGE_NAME) $(DOCKER_BUILD_ARGS) -f docker/tpu.Dockerfile .

.PHONY: docker_build_local
docker_build_local:
	sudo docker build -t $(DOCKER_IMAGE_NAME) $(DOCKER_BUILD_ARGS) -f docker/local.Dockerfile .

.PHONY: docker_run
docker_run:
	sudo docker run $(DOCKER_RUN_FLAGS) --name $(DOCKER_CONTAINER_NAME) -v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME) $(command)

.PHONY: docker_start
docker_start:
	sudo docker run -itd $(DOCKER_RUN_FLAGS) --name $(DOCKER_CONTAINER_NAME) -v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME)

.PHONY: docker_enter
docker_enter:
	sudo docker exec -it $(DOCKER_CONTAINER_NAME) /bin/bash

.PHONY: docker_interactive
docker_interactive: docker_start docker_enter
	echo "Running container $(DOCKER_CONTAINER_NAME) in interactive mode."

.PHONY: docker_kill
docker_kill:
	sudo docker kill $(DOCKER_CONTAINER_NAME)

.PHONY: docker_list
docker_list:
	sudo docker ps

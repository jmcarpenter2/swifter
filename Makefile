.PHONY: help, build, ci-black, ci-flake8, ci-unittest, ci, black, sphinx, git-tag, release-tag, release-production, dev-start, dev-stop

IMAGE=swifter
VERSION_FILE:=swifter/__init__.py
VERSION_TAG:=$(shell cat ${VERSION_FILE} | grep -oEi [0-9]+.[0-9]+.[0-9]+)
TAG?=test

help: ## Display help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

build: ## Build docker image with tag "latest" for unit testing
	DOCKER_BUILDKIT=1 docker build --build-arg GRANT_SUDO="yes" -t $(IMAGE) . -f docker/Dockerfile-dev

release-tag: ## TOODO
	echo ${TAG}

release-production: #TODO
	echo ${VERSION_TAG}

git-tag: ## Tag in git from VERSION file then push tag up to origin
	git tag $(VERSION_TAG)
	git push origin $(VERSION_TAG)

ci-black: build ## Test for black requirements
	docker run --rm -t -v ${PWD}:/mnt $(IMAGE) black --diff --color --check swifter

ci-flake8: build ## Test for flake8 requirements
	docker run --rm -v ${PWD}:/mnt -t $(IMAGE) flake8 swifter

ci-unittest: build ## Test pytest unittests
	docker run --rm -v ${PWD}:/mnt -t $(IMAGE) python -m unittest swifter/swifter_tests.py

ci: ci-black ci-flake8 ci-unittest ## Check black, flake8, and unittests
	@echo "CI successful"

black: build ## Run black, which formats code
	docker run --rm -v ${PWD}:/mnt -t $(IMAGE) black swifter

dev-start: ## Primary make command for dev, spins up containers
	docker-compose -f docker/docker-compose.yml --project-name swifter up -d --build

dev-stop: dev-start ## Spins down active containers
	docker-compose -f docker/docker-compose.yml --project-name swifter down

sphinx: ## Creates docs using sphinx
	echo "Not implemented"

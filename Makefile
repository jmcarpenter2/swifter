.PHONY: help, build, ci-black, ci-flake8, ci-unittest, ci, black, sphinx, git-tag, release-tag, release-production, dev-start, dev-stop

PROJECT=swifter
IMAGE=${PROJECT}
VERSION_FILE:=${PROJECT}/__init__.py
VERSION_TAG:=$(shell cat ${VERSION_FILE} | grep -oEi [0-9]+.[0-9]+.[0-9]+)

help: ## Display help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

build: ## Build docker image with tag "latest" for unit testing
	cd docker && DOCKER_BUILDKIT=1 docker build --build-arg GRANT_SUDO="yes" -t $(IMAGE) . -f Dockerfile-dev && cd ..

sha256: ## Get the openssl sha256 of the deployed version
	curl -sL https://url.com/you/want/to/get.tar.gz | openssl sha256

git-tag: ## Tag in git from VERSION file then push tag up to origin
	echo $(VERSION_TAG)
	git tag $(VERSION_TAG)
	git push --tag

release-production-pypi: ## Builds and publishes the version to PyPi
	python3 setup.py build sdist
	twine upload dist/swifter-${VERSION_TAG}.tar.gz
	rm -r dist && rm -r build

release-production: git-tag release-production-pypi

ci-black: build ## Test for black requirements
	docker run --rm -t -v ${PWD}:/mnt $(IMAGE) black --line-length 120 --diff --color --check ${PROJECT}

ci-flake8: build ## Test for flake8 requirements
	docker run --rm -v ${PWD}:/mnt -t $(IMAGE) flake8 ${PROJECT}

ci-unittest: build ## Test pytest unittests
	docker run --rm -v ${PWD}:/mnt -t $(IMAGE) python -m unittest ${PROJECT}/swifter_tests.py

ci: ci-black ci-flake8 ci-unittest ## Check black, flake8, and unittests
	@echo "CI successful"

black: build ## Run black, which formats code
	docker run --rm -v ${PWD}:/mnt -t $(IMAGE) black --line-length 120 ${PROJECT}

dev-start: ## Primary make command for dev, spins up containers
	docker-compose -f docker/docker-compose.yml --project-name ${PROJECT} up -d --build

dev-stop: dev-start ## Spins down active containers
	docker-compose -f docker/docker-compose.yml --project-name ${PROJECT} down

sphinx: ## Creates docs using sphinx
	echo "Not implemented"

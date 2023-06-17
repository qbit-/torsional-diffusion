# bring in the micromamba image so we can copy files from it
ARG MICROMAMBA_IMAGE=mambaorg/micromamba:1.4.3
# This is the image we are going add micromaba to:
ARG BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
FROM ${MICROMAMBA_IMAGE} as micromamba
FROM ${BASE_IMAGE}

# set up id/gid and micromamba locations
USER root
ARG MAMBA_USER=mamba
ARG MAMBA_USER_ID=1005
ARG MAMBA_USER_GID=1005
ENV MAMBA_USER=$MAMBA_USER
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

# copy micromamba scripts
COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

# create user accounts and the root prefix folder
RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh
RUN chown -R $MAMBA_USER_GID:$MAMBA_USER_ID "$MAMBA_ROOT_PREFIX"

# activate current env in the docker shell
ARG MAMBA_DOCKERFILE_ACTIVATE=1
USER $MAMBA_USER
SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

# add shell initialization to /home/$MAMBA_USER/.bashrc, move activation cmd to
# the end of the script
RUN micromamba shell init --shell=bash --prefix=$MAMBA_ROOT_PREFIX
RUN grep --invert-match "source /usr/local/bin/_activate_current_env.sh" \
    /home/$MAMBA_USER/.bashrc > /tmp/.bashrc && \
    cp /tmp/.bashrc /home/$MAMBA_USER/.bashrc && \
    echo "source /usr/local/bin/_activate_current_env.sh" >> /home/$MAMBA_USER/.bashrc

# install packages
COPY --chown=mamba:mamba environment.lock /tmp/env.lock
RUN micromamba install -y -f /tmp/env.lock && \
   micromamba clean --all --yes

# add script to infinitely spin the docker container
#COPY --chown=mamba:mamba infinite_sleep.sh /usr/local/bin/infinite_sleep.sh
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
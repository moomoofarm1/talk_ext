# copied and modified from tensorflow::install.R, https://github.com/rstudio/tensorflow/blob/master/R/install.R
# and https://github.com/quanteda/spacyr/tree/master/R

conda_args <- reticulate:::conda_args

#' Install talk required python packages in conda or virtualenv environment
#'
#' @description Install talk required python packages (rpp) in a self-contained environment.
#' For macOS and Linux-based systems, this will also install Python itself via a "miniconda" environment, for
#'   \code{talkrpp_install}.  Alternatively, an existing conda installation may be
#'   used, by specifying its path.  The default setting of \code{"auto"} will
#'   locate and use an existing installation automatically, or download and
#'   install one if none exists.
#'
#'   For Windows, automatic installation of miniconda installation is not currently
#'   available, so the user will need to install
#'   \href{https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html}{miniconda
#'    (or Anaconda) manually}.
#' @param conda character; path to conda executable. Default "auto" which
#'   automatically find the path
#' @param update_conda Boolean; update to the latest version of Miniconda after install?
#' (should be combined with force_conda = TRUE)
#' @param force_conda Boolean; force re-installation if Miniconda is already installed at the requested path?
#' @param pip \code{TRUE} to use pip for installing rpp If \code{FALSE}, conda
#' package manager with conda-forge channel will be used for installing rpp.
#' @param rpp_version Character. Specifies the version set of Python packages to install.
#' The default is "rpp_version_system_specific_defaults", which automatically selects compatible
#' Python and package versions based on the user's operating system (e.g., macOS, Windows, Linux).
#'
#' You may also provide a custom character vector of specific package versions
#' (e.g., c("torch==2.0.0", "transformers==4.19.2")) for advanced use cases.
#'
#' Alternatively, set to "talk_diarize" to install a fixed environment required for the diarization
#' functionality, which currently depends on a specific combination of package versions.
#' @param python_version character; default is "python_version_system_specific_defaults". You can specify your
#' Python version for the condaenv yourself.
#'   installation.
#' @param python_path character; path to Python only for virtualenvironment installation
#' @param bin character; e.g., "python", only for virtualenvironment installation
#' @param envname character; name of the conda-environment to install talk required python packages.
#'   Default is "talkrpp_condaenv".
#' @param prompt logical; ask whether to proceed during the installation
#' @examples
#' \dontrun{
#' # install talk required python packages in a miniconda environment (macOS and Linux)
#' talkrpp_install(prompt = FALSE)
#'
#' # install talk required python packages to an existing conda environment
#' talkrpp_install(conda = "~/anaconda/bin/")
#' }
#' @export
talkrpp_install <- function(
    conda = "auto",
    update_conda = FALSE,
    force_conda = FALSE,
    rpp_version = "rpp_version_system_specific_defaults",
    python_version = "python_version_system_specific_defaults",
    envname = "talkrpp_condaenv",
    pip = TRUE,
    python_path = NULL,
    prompt = TRUE) {

  # Set system specific default versions
  if (rpp_version[[1]] == "talk_diarize") {
    rpp_version <- c(

      #### for Nemo
      "cython==3.0.11",
      "wget==3.2",
      "nemo_toolkit==1.20.0",
      "git+https://github.com/m-bain/whisperX.git@78dcfaab51005aa703ee21375f81ed31bc248560",
      "git+https://github.com/adefossez/demucs.git@b9ab48cad45976ba42b2ff17b229c071f0df9390",
      "git+https://github.com/oliverguhr/deepmultilingualpunctuation.git@5a0dd7f4fd56687f59405aa8eba1144393d8b74b",
      "git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git@abd458dd879305566cd4ed0c8624c95f22e3126a",
      "huggingface-hub==0.23.2",
      #     "ctranslate2==3.24.0",
      "PyYAML==6.0.2",
      "hydra-core==1.3.2",
      "youtokentome==1.0.5", # 1.0.6
      "inflect==7.4.0",
      "webdataset==0.2.100",
      "editdistance==0.8.1",
      "jiwer==3.0.5",
      ###        "ffmpeg==1.4", instead: brew install ffmpeg
      "pytorch-lightning==1.9.4",
      "ipython==8.31.0",
      "ffmpeg-python==0.2.0",
      "librosa",
      "torchaudio",
      "llvmlite==0.40.0",
      "numba==0.57.0"
    )

  }

  if (rpp_version[[1]] == "rpp_version_system_specific_defaults") {
    if (is_osx() || is_linux()) {
      rpp_version <- c(
        # talk
        "numpy==1.26.0",
        "torch==2.2.0",
        "torchaudio==2.2.0",
        "librosa==0.10.2",
        "soundfile==0.12.1",
        "transformers==4.38.0",
        "huggingface_hub==0.20.0",
#
        "argparse",
        "pandas",
        "tqdm",
        "matplotlib",
        "ffmpeg-python"
      )
    }
    if (is_windows()) {
      rpp_version <- c(
        # talk
        "numpy==1.26.0",
        "torch==2.2.0",
        "torchaudio==2.2.0",
        "librosa==0.10.2",
        "soundfile==0.12.1",
        "transformers==4.38.0",
        "huggingface_hub==0.20.0",

        "argparse",
        "pandas",
        "tqdm",
        "matplotlib",
        "ffmpeg-python"
      )
    }
  }

  if (python_version == "python_version_system_specific_defaults") {
    if (is_osx() || is_linux()) {
      python_version <- "3.10.12"
    }

    if (is_windows()) {
      python_version <- "3.10.12"
    }
  }

  # verify os
  if (!is_windows() && !is_osx() && !is_linux()) {
    stop("This function is available only for Windows, Mac, and Linux")
  }

  # verify 64-bit
  if (.Machine$sizeof.pointer != 8) {
    stop(
      "Unable to install the talk-package on this platform.",
      "Binary installation is only available for 64-bit platforms."
    )
  }

  # install rust for singularity machine -- but it gives error in github action
  # reticulate::py_run_string("import os\nos.system(\"curl --proto '=https' --tlsv1.2 -sSf
  # https://sh.rustup.rs | sh -s -- -y\")")
  system("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")

  # resolve and look for conda help(conda_binary)
  conda <- tryCatch(reticulate::conda_binary(conda), error = function(e) NULL)
  have_conda <- !is.null(conda)

  # Mac and linux
  if (is_unix()) {
    # check for explicit conda method
    # validate that we have conda
    if (!have_conda) {
      message("No conda was found in the system. ")
      if (prompt) {
        ans <- utils::menu(c("No", "Yes"), title = "Do you want talk to download
                           miniconda using reticulate::install_miniconda()?")
      } else {
        ans <- 2 # When no prompt is set to false, default to install miniconda.
      }
      if (ans == 2) {
        reticulate::install_miniconda(update = update_conda)
        conda <- tryCatch(reticulate::conda_binary("auto"), error = function(e) NULL)
      } else {
        stop("Conda environment installation failed (no conda binary found)\n", call. = FALSE)
      }
    }

    # Update mini_conda
    if (update_conda && force_conda || force_conda) {
      reticulate::install_miniconda(update = update_conda, force = force_conda)
    }

    # process the installation of talk required python packages
    process_talkrpp_installation_conda(conda,
      rpp_version,
      python_version,
      prompt,
      envname = envname,
      pip = pip
    )

    # Windows installation
  } else {
    # determine whether we have system python help(py_versions_windows)
    if (python_version == "find_python") {
      python_versions <- reticulate::py_versions_windows()
      python_versions <- python_versions[python_versions$type == "PythonCore", ]
      python_versions <- python_versions[python_versions$version %in% c("3.5", "3.6", "3.7", "3.8", "3.9"), ]
      python_versions <- python_versions[python_versions$arch == "x64", ]
      have_system <- nrow(python_versions) > 0

      if (have_system) {
        # Well this isn't used later
        python_version <- python_versions[1, ]
      }
    }

    # validate that we have conda:
    if (!have_conda) {
      # OK adds help(install_miniconda)
      reticulate::install_miniconda(update = update_conda)
      conda <- tryCatch(reticulate::conda_binary("auto"), error = function(e) NULL)
    }
    # Update mini_conda
    if (have_conda && update_conda || have_conda && force_conda) {
      reticulate::install_miniconda(update = update_conda, force = force_conda)
    }
    # process the installation of talk required python packages
    process_talkrpp_installation_conda(conda,
      rpp_version,
      python_version,
      prompt,
      envname = envname,
      pip = pip
    )
  }

  message(colourise(
    "\nInstallation is completed.\n",
    fg = "blue", bg = NULL
  ))
  message(
    " ",
    sprintf("Condaenv: %s ", envname), "\n"
  )

  message(colourise(
    "Great work - do not forget to initialize the environment \nwith talkrpp_initialize().\n",
    fg = "green", bg = NULL
  ))
  invisible(NULL)
}

process_talkrpp_installation_conda <- function(conda,
                                               rpp_version,
                                               python_version,
                                               prompt = TRUE,
                                               envname = "talkrpp_condaenv",
                                               pip = FALSE) {
  conda_envs <- reticulate::conda_list(conda = conda)
  if (prompt) {
    ans <- utils::menu(c("Confirm", "Cancel"), title = "Confirm that a new conda environment will be set up.")
    if (ans == 2) stop("condaenv setup is cancelled by user", call. = FALSE)
  }
  conda_env <- subset(conda_envs, conda_envs$name == envname)
  if (nrow(conda_env) == 1) {
    message(
      "Using existing conda environment ", envname, " for talk installation\n.",
      "\ntalk:",
      paste(rpp_version, collapse = ", "), "will be installed.  "
    )
  } else {
    message(
      "A new conda environment", paste0('"', envname, '"'), "will be created and \npython required packages:",
      paste(rpp_version, collapse = ", "), "will be installed.  "
    )
    message("Creating", envname, "conda environment for talk installation...\n")
    python_packages <- ifelse(is.null(python_version), "python=3.9",
      sprintf("python=%s", python_version)
    )
    python <- reticulate::conda_create(envname, packages = python_packages, conda = conda)
  }

  message("Installing talk required python packages...\n")
  packages <- rpp_version

  reticulate::conda_install(envname, packages, pip = pip, conda = conda)
}



process_talkrpp_installation_virtualenv <- function(python = "/usr/local/bin/python3.9",
                                                    rpp_version,
                                                    pip_version,
                                                    envname = "talkrpp_virtualenv",
                                                    prompt = TRUE) {
  libraries <- paste(rpp_version, collapse = ", ")
  message(sprintf(
    'A new virtual environment called "%s" will be created using "%s" \n and,
    the following talk reuired python packages will be installed: \n "%s" \n \n',
    envname, python, libraries
  ))
  if (prompt) {
    ans <- utils::menu(c("No", "Yes"), title = "Proceed?")
    if (ans == 1) stop("Virtualenv setup is cancelled by user", call. = FALSE)
  }

  # Make python path help(virtualenv_create)
  reticulate::virtualenv_create(envname,
                                python,
                                pip_version = NULL,
                                required = TRUE)

  reticulate::use_virtualenv(envname, required = TRUE)

  #
  for (i in seq_len(length(rpp_version))) {
    reticulate::py_install(rpp_version[[i]], envname = envname, pip = TRUE)
  }

  message(colourise(
    "\nSuccess!\n",
    fg = "green", bg = NULL
  ))
}

# Check whether "bin"/something exists in the bin folder
# For example, bin = "pip3" bin = "python3.9" bin = ".virtualenv"
# And for example: file.exists("/usr/local/bin/.virtualenvs") /Users/oscarkjell/.virtualenvs
python_unix_binary <- function(bin) {
  locations <- file.path(c("/usr/local/bin", "/usr/bin"), bin)
  locations <- locations[file.exists(locations)]
  if (length(locations) > 0) {
    locations[[1]]
  } else {
    NULL
  }
}

#' @rdname talkrpp_install
#' @description If you wish to install Python in a "virtualenv", use the
#'   \code{talkrpp_install_virtualenv} function. It requires that you have a python version
#'   and path to it (such as "/usr/local/bin/python3.9" for Mac and Linux.).
#' @param pip_version character;
#' @examples
#' \dontrun{
#' # install talk required python packages in a virtual environment
#' talkrpp_install_virtualenv()
#' }
#' @export
talkrpp_install_virtualenv <- function(rpp_version = c("torch==2.0.0",
                                                       "transformers==4.19.2",
                                                       "numpy",
                                                       "pandas",
                                                       "nltk"),
                                       python_path = NULL, # "/usr/local/bin/python3.9",
                                       pip_version = NULL,
                                       bin = "python3",
                                       envname = "talkrpp_virtualenv",
                                       prompt = TRUE) {
  # find system python binary
  if (!is.null(python_path)) {
    python <- python_path
    } else {
      python <-  python_unix_binary(bin = bin)
    }


  if (is.null(python)) {
    stop("Unable to locate Python on this system.", call. = FALSE)
  }

  process_talkrpp_installation_virtualenv(
    python = python,
    pip_version = pip_version,
    rpp_version = rpp_version,
    envname = envname,
    prompt = prompt
  )


  message(colourise(
    "\nInstallation is completed.\n",
    fg = "blue", bg = NULL
  ))
  invisible(NULL)
}


#' Uninstall talkrpp conda environment
#'
#' Removes the conda environment created by talkrpp_install()
#' @param conda path to conda executable, default to "auto" which automatically
#'   finds the path
#' @param prompt logical; ask whether to proceed during the installation
#' @param envname character; name of conda environment to remove
#' @export
talkrpp_uninstall <- function(conda = "auto",
                              prompt = TRUE,
                              envname = "talkrpp_condaenv") {
  conda <- tryCatch(reticulate::conda_binary(conda), error = function(e) NULL)
  have_conda <- !is.null(conda)

  if (!have_conda) {
    stop("Conda installation failed (no conda binary found)\n", call. = FALSE)
  }

  conda_envs <- reticulate::conda_list(conda = conda)
  conda_env <- subset(conda_envs, conda_envs$name == envname)
  if (nrow(conda_env) != 1) {
    stop("conda environment", envname, "is not found", call. = FALSE)
  }
  message("A conda environment", envname, "will be removed\n")
  ans <- ifelse(prompt, utils::menu(c("No", "Yes"), title = "Proceed?"), 2)
  if (ans == 1) stop("condaenv removal is cancelled by user", call. = FALSE)
  python <- reticulate::conda_remove(envname = envname)

  message("\nUninstallation complete.\n\n")

  invisible(NULL)
}

###### see utils.R in spacyr
# checking OS functions, thanks to r-tensorflow;

is_windows <- function() {
  identical(.Platform$OS.type, "windows")
}

is_unix <- function() {
  identical(.Platform$OS.type, "unix")
}

is_osx <- function() {
  Sys.info()["sysname"] == "Darwin"
}

is_linux <- function() {
  identical(tolower(Sys.info()[["sysname"]]), "linux")
}

#is_ubuntu <- function() {
#  if (is_unix() && file.exists("/etc/lsb-release")) {
#    lsbrelease <- readLines("/etc/lsb-release")
#    any(grepl("Ubuntu", lsbrelease))
#  } else {
#    FALSE
#  }
#}

#python_version_function <- function(python) {
#  # check for the version
#  result <- system2(python, "--version", stdout = TRUE, stderr = TRUE)
#
#  # check for error
#  error_status <- attr(result, "status")
#  if (!is.null(error_status)) {
#    stop("Error ", error_status, " occurred while checking for python version", call. = FALSE)
#  }
#
#  # parse out the major and minor version numbers
#  matches <- regexec("^[^ ]+\\s+(\\d+)\\.(\\d+).*$", result)
#  matches <- regmatches(result, matches)[[1]]
#  if (length(matches) != 3) {
#    stop("Unable to parse Python version '", result[[1]], "'", call. = FALSE)
#  }
#
#  # return as R numeric version
#  numeric_version(paste(matches[[2]], matches[[3]], sep = "."))
#}

#pip_get_version <- function(cmd, major_version) {
#  regex <- "^(\\S+)\\s?(.*)$"
#  cmd1 <- sub(regex, "\\1", cmd)
#  cmd2 <- sub(regex, "\\2", cmd)
#  oldw <- getOption("warn")
#  options(warn = -1)
#  result <- paste(system2(cmd1, cmd2, stdout = TRUE, stderr = TRUE),
#    collapse = " "
#  )
#  options(warn = oldw)
#  version_check_regex <- sprintf(".+(%s.\\d+\\.\\d+).+", major_version)
#  return(sub(version_check_regex, "\\1", result))
#}


#conda_get_version <- function(major_version = NA, conda, envname) {
#  condaenv_bin <- function(bin) path.expand(file.path(dirname(conda), bin))
#  cmd <- sprintf(
#    "%s%s %s && conda search torch -c conda-forge%s",
#    ifelse(is_windows(), "", ifelse(is_osx(), "source ", "/bin/bash -c \"source ")),
#    shQuote(path.expand(condaenv_bin("activate"))),
#    envname,
#    ifelse(is_windows(), "", ifelse(is_osx(), "", "\""))
#  )
#  regex <- "^(\\S+)\\s?(.*)$"
#  cmd1 <- sub(regex, "\\1", cmd)
#  cmd2 <- sub(regex, "\\2", cmd)
#
#  result <- system2(cmd1, cmd2, stdout = TRUE, stderr = TRUE)
#  result <- sub("\\S+\\s+(\\S+)\\s.+", "\\1", result)
#  if (!is.na(major_version)) {
#    result <- grep(paste0("^", major_version, "\\."), result, value = TRUE)
#  }
#  #
#  return(result[length(result)])
#}



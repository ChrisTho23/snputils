import inspect
import os
import pathlib
import subprocess
import urllib.request
import zipfile
import platform

import pytest

import snputils
from snputils.snp.io.read import BEDReader, PGENReader, VCFReader


@pytest.fixture(scope="module")
def data_path():
    module_path = pathlib.Path(inspect.getfile(snputils)).parent.parent
    data_path = module_path / "data"
    os.makedirs(data_path, exist_ok=True)

    system = platform.system()
    is_arm = platform.machine().lower().startswith(('arm', 'aarch'))

    plink_urls = {
        ("Darwin", True): "plink2_mac_arm64_20241114.zip",  # macOS ARM
        ("Linux", False): "plink2_linux_x86_64_20241114.zip",  # Linux x86_64
    }

    try:
        plink_filename = plink_urls[(system, is_arm)]
    except KeyError:
        raise RuntimeError(f"Unsupported platform: {system} {platform.machine()}")

    files_urls = {
        plink_filename: f"https://s3.amazonaws.com/plink2-assets/alpha6/{plink_filename}",
        "vcf/ALL.chr22.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz":
        "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/20181203_biallelic_SNV/ALL.chr22.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz",
    }

    # Check and download each file if it does not exist
    for file_name, url in files_urls.items():
        file_path = data_path / file_name
        dir_path = file_path.parent if file_path.is_file() else file_path
        if not dir_path.exists():
            os.makedirs(file_path.parent, exist_ok=True)
        if not file_path.exists():
            print(f"Downloading {file_name} to {data_path}. This may take a while...")
            urllib.request.urlretrieve(url, file_path)
            if file_path.suffix == ".zip":
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(data_path)
                subprocess.run(
                    ["chmod", "+x", data_path / "plink2"], cwd=str(data_path)
                )

        if file_path.suffixes[-2:] == ['.vcf', '.gz']:
            # Subset sample files
            subset_file = data_path / "subset.txt"
            if not subset_file.exists():
                print("Generating subset file...")
                four_sample_ids = ["HG00096", "HG00097", "HG00099", "HG00100"]
                with open(subset_file, "w") as file:
                    file.write("\n".join(four_sample_ids))

            plink_vcf_out = "vcf/subset"
            subset_vcf_file = data_path / (plink_vcf_out + ".vcf")
            if not subset_vcf_file.exists():
                print("Generating subset VCF...")
                subprocess.run(
                    [
                        "./plink2",
                        "--vcf",
                        data_path / file_name,
                        "--keep",
                        "subset.txt",
                        "--recode",
                        "vcf",
                        "--set-missing-var-ids",
                        "@:#",
                        "--out",
                        plink_vcf_out,
                    ],
                    cwd=str(data_path),
                )

    # Generate bed and pgen formats
    for fmt in ["bed", "pgen", "pgen_zst"]:
        fmt_path = data_path / fmt
        os.makedirs(fmt_path, exist_ok=True)
        fmt_file = fmt_path / "subset"
        if not fmt_file.exists():
            print(f"Generating {fmt} format...")
            make_fmt = "--make-pgen vzs" if fmt == "pgen_zst" else f"--make-{fmt}"
            subprocess.run(
                [
                    "./plink2",
                    "--vcf",
                    subset_vcf_file,
                    *make_fmt.split(),
                    "--out",
                    fmt_file,
                ],
                cwd=str(data_path),
            )

    return str(data_path)


@pytest.fixture(scope="module")
def snpobj_vcf(data_path):
    return VCFReader(data_path + "/vcf/subset.vcf").read(sum_strands=False)


@pytest.fixture(scope="module")
def snpobj_bed(data_path):
    return BEDReader(data_path + "/bed/subset").read(sum_strands=False)


@pytest.fixture(scope="module")
def snpobj_pgen(data_path):
    return PGENReader(data_path + "/pgen/subset").read(sum_strands=False)

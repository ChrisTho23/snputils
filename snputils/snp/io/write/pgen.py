import logging
import numpy as np
import polars as pl
import pgenlib as pg
from pathlib import Path
import zstandard as zstd

from snputils.snp.genobj.snpobj import SNPObject

log = logging.getLogger(__name__)


class PGENWriter:
    """
    Writes a genotype object in PGEN format (.pgen, .psam, and .pvar files) in the specified output path.
    """

    def __init__(self, snpobj: SNPObject, filename: str):
        """
        Initializes the PGENWriter instance.

        Args:
            snpobj (SNPObject): The SNPObject containing genotype data to be written.
            filename (str): Base path for the output files (excluding extension).
        """
        self.__snpobj = snpobj
        self.__filename = Path(filename)

    def write(self, vzs: bool = False):
        """
        Writes the SNPObject data to .pgen, .psam, and .pvar files.

        Args:
            vzs (bool, optional): If True, compresses the .pvar file using zstd and saves it as .pvar.zst. Defaults to False.
        """
        file_extensions = (".pgen", ".psam", ".pvar", ".pvar.zst")
        if self.__filename.suffix in file_extensions:
            self.__filename = self.__filename.with_suffix('')
        self.__file_extension = ".pgen"

        self.write_pvar(vzs=vzs)
        self.write_psam()
        self.write_pgen()

    def write_pvar(self, vzs: bool = False):
        """
        Writes variant data to the .pvar file.

        Args:
            vzs (bool, optional): If True, compresses the .pvar file using zstd and saves it as .pvar.zst. Defaults to False.
        """
        output_filename = f"{self.__filename}.pvar"
        if vzs:
            output_filename += ".zst"
            log.info(f"Writing to {output_filename} (compressed)")
        else:
            log.info(f"Writing to {output_filename}")

        df = pl.DataFrame(
            {
                "#CHROM": self.__snpobj.variants_chrom,
                "POS": self.__snpobj.variants_pos,
                "ID": self.__snpobj.variants_id,
                "REF": self.__snpobj.variants_ref,
                "ALT": self.__snpobj.variants_alt[:, 0],
                "FILTER": self.__snpobj.variants_filter_pass,
                # TODO: add INFO column to SNPObject and write it to the .pvar file? (if not it's lost)
            }
        )
        # TODO: add header to the .pvar file, if not it's lost

        # Write the DataFrame to a CSV string
        csv_data = df.write_csv(None, separator="\t")

        if vzs:
            # Compress the CSV data using zstd
            cctx = zstd.ZstdCompressor()
            compressed_data = cctx.compress(csv_data.encode('utf-8'))
            with open(output_filename, 'wb') as f:
                f.write(compressed_data)
        else:
            with open(output_filename, 'w') as f:
                f.write(csv_data)

    def write_psam(self):
        """
        Writes sample metadata to the .psam file.
        """
        log.info(f"Writing {self.__filename}.psam")
        df = pl.DataFrame(
            {
                "IID": self.__snpobj.samples,
                "SEX": "NA",  # Add SEX as nan for now
                # TODO: add SEX as Optional column to SNPObject and write it to the .psam file (if not it's lost)
            }
        )
        df.write_csv(f"{self.__filename}.psam", separator="\t")

    def write_pgen(self):
        """
        Writes the genotype data to a .pgen file.
        """
        log.info(f"Writing to {self.__filename}.pgen")
        phased = True if self.__snpobj.calldata_gt.ndim == 3 else False
        if phased:
            num_variants, num_samples, num_alleles = self.__snpobj.calldata_gt.shape
            # Flatten the genotype matrix for pgenlib
            flat_genotypes = self.__snpobj.calldata_gt.reshape(
                num_variants, num_samples * num_alleles
            )
        else:
            num_variants, num_samples = self.__snpobj.calldata_gt.shape
            flat_genotypes = self.__snpobj.calldata_gt

        with pg.PgenWriter(
            filename=str(self.__filename).encode('utf-8'),
            sample_ct=num_samples,
            variant_ct=num_variants,
            hardcall_phase_present=phased,
        ) as writer:
            for variant_index in range(num_variants):
                writer.append_alleles(
                    flat_genotypes[variant_index].astype(np.int32), all_phased=phased
                )

"""
Copyright (c) 2020-2022 Remi Cresson (INRAE)

Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import argparse
import otbApplication
import logging

from sr4rs import constants

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.WARNING,
                    datefmt='%Y-%m-%d %H:%M:%S')

# Available encodings in OTB
encodings = {"unsigned_char": otbApplication.ImagePixelType_uint8,
             "short": otbApplication.ImagePixelType_int16,
             "unsigned_short": otbApplication.ImagePixelType_uint16,
             "int": otbApplication.ImagePixelType_int32,
             "unsigned_int": otbApplication.ImagePixelType_uint32,
             "float": otbApplication.ImagePixelType_float,
             "double": otbApplication.ImagePixelType_double}

DEFAULT_PAD = 64
DEFAULT_ENCODING = "auto"
DEFAULT_TILE_SIZE = 512


def get_encoding_name(input):
    """
    Get the encoding of input image pixels
    """
    infos = otbApplication.Registry.CreateApplication('ReadImageInfo')
    infos.SetParameterString("in", input)
    infos.Execute()
    return infos.GetParameterString("datatype")


def run(input, output, saved_model, pad=DEFAULT_PAD, ts=DEFAULT_TILE_SIZE, encoding=DEFAULT_ENCODING):
    gen_fcn = pad  # Available shrinked outputs
    efield = ts  # OTBTF expression field
    if efield % min(constants.factors) != 0:
        logging.fatal("Please chose a tile size that is consistent with the network.")
        quit()
    ratio = 1.0 / float(max(constants.factors))  # OTBTF Spacing ratio
    rfield = int((efield + 2 * gen_fcn) * ratio)  # OTBTF receptive field
    # pixel encoding
    encoding_name = get_encoding_name(input) if encoding == "auto" else encoding
    encoding = encodings[encoding_name]
    logging.info("Using encoding %s", encoding)
    # call otbtf
    logging.info("Receptive field: {}, Expression field: {}".format(rfield, efield))
    ph = "{}{}".format(constants.outputs_prefix, pad)
    infer = otbApplication.Registry.CreateApplication("TensorflowModelServe")
    infer.SetParameterStringList("source1.il", [input])
    infer.SetParameterInt("source1.rfieldx", rfield)
    infer.SetParameterInt("source1.rfieldy", rfield)
    infer.SetParameterString("source1.placeholder", constants.lr_input_name)
    infer.SetParameterString("model.dir", saved_model)
    infer.SetParameterString("model.fullyconv", "on")
    infer.SetParameterStringList("output.names", [ph])
    infer.SetParameterInt("output.efieldx", efield)
    infer.SetParameterInt("output.efieldy", efield)
    infer.SetParameterFloat("output.spcscale", ratio)
    infer.SetParameterInt("optim.tilesizex", efield)
    infer.SetParameterInt("optim.tilesizey", efield)
    infer.SetParameterInt("optim.disabletiling", 1)
    out_fn = "{}{}".format(output, "?" if "?" not in output else "")
    out_fn += "&streaming:type=tiled&streaming:sizemode=height&streaming:sizevalue={}".format(efield)
    infer.SetParameterString("out", out_fn)
    infer.SetParameterOutputImagePixelType("out", encoding)
    infer.ExecuteAndWriteOutput()


def cli():
    # Arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input LR image. Must be in the same dynamic as the lr_patches used in the "
                                        "train.py application.", required=True)
    parser.add_argument("--savedmodel", help="Input SavedModel (provide the path to the folder).", required=True)
    parser.add_argument("--output", help="Output HR image", required=True)
    parser.add_argument('--encoding', type=str, default=DEFAULT_ENCODING, const=DEFAULT_ENCODING, nargs="?",
                        choices=encodings.keys(),
                        help="Output HR image encoding")
    parser.add_argument('--pad', type=int, default=DEFAULT_PAD, const=DEFAULT_PAD, nargs="?",
                        choices=constants.pads,
                        help="Margin size for blocking artefacts removal")
    parser.add_argument('--ts', default=DEFAULT_TILE_SIZE, type=int,
                        help="Tile size. Tune this to process larger output image chunks, "
                             "and speed up the process.")
    params = parser.parse_args()
    run(params.pad, params.ts, params.encoding, params.savedmodel, params.input, params.output)


if __name__ == "__main__":
    cli()

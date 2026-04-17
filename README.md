# wd14-tagger

An image tagger using the `SmilingWolf/wd-vit-large-tagger-v3` image tagging model written in rust.

Usage: wd14-tagger [OPTIONS] <IMAGE>

```
Arguments:
  <IMAGE>  Path to the image file

Options:
  -o, --output <OUTPUT>  Optional output file to write results
  -m, --model <MODEL>    Model backend to use [default: wd14] [possible values: wd14, dino]
  -h, --help             Print help
```

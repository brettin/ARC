# EpiHiper Schema

## Installation

``` sh
npm install @shoops/epi-hiper-validator
```

## Usage
After installation you should have access to two command line tools `epiHiperValidateSchema` and `epiHiperValidateRules`. If you cannot execute either of them, you will have the directory `./node_modules/.bin` to your `PATH` environment variable or access these tools by their full path.

### `epiHiperValidateSchema`
This command line tool allows you validate one or more files against the EpiHiper [schemas](./schema). The schema is automatically determined by the `$schema` attribute of the root JSON object. For validation against externally provided schemas specify the schema containing directory with the `--schema-dir` options.
``` 
Usage: epiHiperValidateSchema [options] <file ...>

EpiHiper Schema Validation

Options:
  -V, --version              output the version number
  --schema-dir <schema-dir>  Use the schemas in the provided directory instead of the internal ones.
  -h, --help                 output usage information
```
### `epiHiperValidateRules`
This command line tool allows you validate one or more files against the EpiHiper [rules](./schema). The rules are automatically determined by the `$schema` attribute of the root JSON object.For validation against externally provided rules specify the rules containing directory with the `--rules-dir` options.
``` 
Usage: epiHiperValidateRules [options] <file ...>

EpiHiper Rules Validation

Options:
  -V, --version            output the version number
  --rules-dir <rules-dir>  Use the rules in the provided directory instead of the internal ones.
  -h, --help               output usage information
```

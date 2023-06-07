# alternative_wordings

## Project setup

* Install Anaconda.
* Install [PyTorch](https://pytorch.org/get-started/locally/)
* Install [spacy](https://spacy.io/usage)
* `python -m spacy download en_core_web_sm`
* `pip install -r requirements.txt`
* `npm install`

You will also have to have fairseq's mBART downloaded and extracted from https://dl.fbaipublicfiles.com/fairseq/models/mbart50/mbart50.ft.nn.tar.gz and put into the same location as models.py.
### Backend

`python app.py`

### Frontend (development)

`npm run serve`

### Compiles and minifies for production
```
npm run build
```

### Customize configuration
See [Configuration Reference](https://cli.vuejs.org/config/).


## Public serving

Add the following to `vue.config.js` under the `devServer` section:

```
        public: 'https://dev1.kenarnold.org/',
        allowedHosts: [
            '.kenarnold.org'
        ],
```

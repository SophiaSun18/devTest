<template>
  <div id="app">
    <!--Header, input text area, and button section-->
    <div class="form-group">
      <h2 style="color: black">Exploring Alternative Wordings</h2>
      <textarea
        id="userenglish"
        name="text"
        v-model="inputText"
        rows="4"
        cols="50"
        required
      >
      </textarea
      ><br /><br />
      <button
        class="continue"
        @click="
          incremental(inputText, '', false);
          showResults = true;
          current_text = inputText;
        "
      >
        Continue
      </button>
    </div>
    <!--Area where you can switch out words in the sentence with most likely alternatives-->
    <div class="focus-sentence">
      <draggable v-model="incrementalData.chunks" @end="checkMove">
        <span class="tooltip" v-for="(chunk, ind) in incrementalData.chunks">
          <span v-if="chunk[1]">
            <span style="background-color: yellow">{{ chunk[0] }}</span>
          </span>
          <span v-else>
            <span>{{ chunk[0] }}</span>
          </span>
        </span>
      </draggable>
    </div>
    <div class="focus-sentence">
      <span class="tooltip" v-for="(words, ind) in word_alts">
        <!--If you click on a word to replace it make the replacement yellow-->
        <span v-if="ind == selectedIdx" style="background-color: yellow"
          >{{ words[0] }}
        </span>
        <span v-else>{{ words[0] }}</span>
        <!--Drop down selector for alternative words from prediction-->
        <div id="ind" class="tooltiptext">
          <span v-for="i in word_alts[ind]">
            <button
              class="plain"
              @click="
                recalculate(i, ind);
                selectedIdx = ind;
              "
            >
              {{ i }}</button
            ><br />
          </span>
        </div>
      </span>
    </div>
  </div>
</template>

<script>
import draggable from "vuedraggable";

export default {
  data() {
    return {
      isShowing: [false],
      inputText: "",
      showResults: true,
      altsData: { alternatives: [], colorCoding: [], test: true },
      colors: [
        "white",
        "#ABE3BB",
        "#E4B0AF",
        "#3DA1B8",
        "#E4E2AF",
        "#B8473D",
        "#B39FCF",
        "#30915F",
      ],
      incrementalData: {
        chunks: [],
      },
      withChangedWord: "",
      prefix: "",
      completions: [],
      differences: [],
      selectedIdx: -1,
      current_text: "",
      word_alts: [],
    };
  },

  components: {
    draggable,
  },

  methods: {
    restOfSet(group) {
      console.log(group);
      return group.slice(1);
    },
    toggleShowing: function (event) {
      var targetId = event.currentTarget.id;
      console.log(targetId);
      if (this.isShowing[targetId]) {
        this.$set(this.isShowing, targetId, false);
      } else {
        {
          this.$set(this.isShowing, targetId, true);
        }
      }
      console.log(this.isShowing[targetId]);
    },
    handle(idx) {
      console.log(idx);
      return this.isShowing[idx];
    },
    isDifferent(string, optionidx, wordidx) {
      // Used to bold different words in alternative sentences

      console.log(string);
      console.log(optionidx);
      console.log(wordidx);
      if (this.differences[optionidx].indexOf(wordidx) > -1) {
        console.log("true");
        return true;
      } else {
        return false;
      }
    },
    async getAlts(inputText) {
      // Gets alternatives for inputText along with chunks for color coding
      // Used with continue button

      var url = new URL("/api/result", window.location);
      var params = {
        english: inputText,
      };
      url.searchParams.append("q", JSON.stringify(params));
      const res = await fetch(url);
      const input = await res.json();
      this.altsData = input;
    },

    async incremental(inputText, prefix, recalculation) {
      // Fills this.incrementalData with inputText, expected machine translation.
      // list of tokens in final sequence, list of top 10 alternative words for each token,
      // and a score for average predictability.

      var url = new URL("/api/incremental", window.location);
      var params = {
        english: inputText,
        prefix: prefix,
        recalculation: recalculation,
      };
      url.searchParams.append("q", JSON.stringify(params));
      const res = await fetch(url);
      const input = await res.json();
      this.incrementalData = input;
      console.log(input);
      // reset selectedIdx in case this was called through selecting alternative word
      this.selectedIdx = -1;
    },

    async recalculate(changedword, index) {
      // Get alternative endings once word is chosen from word probability list

      this.$set(this.incrementalData.tokens, index, changedword);
      // get sentence up to and including changed word
      var thelist = this.incrementalData.tokens.slice(0, index + 1);
      var newinputstr = thelist.join("").replace(/\u00a0/g, " ");
      // set as prefix
      this.prefix = newinputstr;
      // this.incremental(this.inputText, newinputstr, true)
      // this.withChangedWord = this.incrementalData.tokens.join('').replace(/\u00a0/g, ' ');

      var url = new URL("/api/completion", window.location);
      var params = {
        sentence: this.inputText,
        prefix: newinputstr,
      };
      url.searchParams.append("q", JSON.stringify(params));
      const res = await fetch(url);
      const input = await res.json();
      let tolist = [];
      var x;
      for (x in input.endings) {
        //split each string into words, preserving spaces
        tolist.push(
          input.endings[x].split(/(\S+\s+)/).filter(function (n) {
            return n;
          })
        );
      }
      this.completions = tolist;
      this.differences = input.differences;

      this.showResults = false;
    },

    fulltext(prefix, completion) {
      return prefix + completion.join("");
    },

    async checkMove(evnt) {
      console.log(evnt);
      var constraints = [];
      var chunk = [];
      for (chunk of this.incrementalData.chunks) {
        if (chunk[1]) {
          //capatalize if first phrase of sentence
          console.log(this.incrementalData.chunks.indexOf(chunk));
          if (this.incrementalData.chunks.indexOf(chunk) === 1) {
            chunk[0] = chunk[0].charAt(0).toUpperCase() + chunk[0].slice(1);
          }
          constraints.push(chunk[0]);
        }
      }
      console.log(this.current_text);
      console.log(constraints);
      var url = new URL("/api/constraints", window.location);
      var params = {
        sentence: this.current_text,
        constraints: constraints,
      };
      url.searchParams.append("q", JSON.stringify(params));
      var res = await fetch(url);
      var output = await res.json();
      var input = output.result;
      this.word_alts = output.word_alternatives;
      console.log(this.word_alts);
      console.log(input);
      url = new URL("/api/incremental", window.location);
      params = {
        english: input,
        prefix: "",
        recalculation: false,
      };
      url.searchParams.append("q", JSON.stringify(params));
      res = await fetch(url);
      input = await res.json();
      console.log(input);
      this.incrementalData = input;
      var current_text = "";
      for (chunk of input.chunks) {
        current_text = current_text.concat(chunk[0]);
      }
      this.current_text = current_text;
    },
  },
};
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}

.diff {
  font-weight: bold;
}

ul {
  line-height: 300%;
}

.form-group {
  background-color: lightgray;
  padding: 3%;
}
.focus-sentence {
  background-color: #eeeeee;
  font-size: 25px;
  padding: 5%;
}
span {
  white-space: pre-wrap;
}
.results {
  text-align: left;
  margin-left: 10%;
  margin-right: 10%;
}
.tooltip {
  position: relative;
  display: inline-block;
  white-space: pre;
}
.tooltip .tooltiptext {
  visibility: hidden;
  /* width: 150px; */
  background-color: white;
  color: white;
  text-align: left;
  border-radius: 6px;
  font-size: 66.67%;
  position: absolute;
  z-index: 1;
}
.tooltiptext {
  padding: 10%;
}
.tooltiptext button.plain {
  font-size: 20px;
  margin: 0.1em;
}
.tooltip:hover .tooltiptext {
  visibility: visible;
}
.tooltip:hover {
  cursor: context-menu;
}

button.plain {
  background: none;
  border: none;
  text-align: left;
  white-space: nowrap;
}
button.plain:hover {
  cursor: pointer;
}

button.plain:focus {
  outline: none;
}

.continue {
  box-shadow: inset 0px 1px 0px 0px white;
  background: linear-gradient(to bottom, white 5%, #f6f6f6 100%);
  background-color: white;
  border-radius: 6px;
  border: 1px solid #dcdcdc;
  display: inline-block;
  color: #666666;
  font-size: 15px;
  font-weight: bold;
  padding: 6px 24px;
  text-shadow: 0px 1px 0px white;
}
.continue:hover {
  background: linear-gradient(to bottom, #f6f6f6 5%, white 100%);
  background-color: #f6f6f6;
  cursor: pointer;
}
.grid-container {
  display: grid;
  grid-template-columns: auto auto auto;
  padding: 10px;
}
.grid-item {
  font-size: 20px;
  text-align: left;
  padding: 0px;
}

.results .tooltip {
  vertical-align: top;
}
</style>

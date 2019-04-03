import {Component, h} from "/common/external/preact.js";
import {Grid} from "/common/lib/mdl.js";
import {stylesheet} from "/common/lib/util.js";

stylesheet("/common/style/docview.css");

let kb_url = '/kb/?id=';

let type_color = {
  '/s': '#FF8000',
  '/saft': '#38761D',
  '/pb': '#990000',
  '/vn': '#8B4513',
  '/w': '#0B5394',
};

let begin_styles = [
  {mask: (1 << 5), tag: "h2"},
  {mask: (1 << 11), tag: "blockquote"},
  {mask: (1 << 7), tag: "ul"},
  {mask: (1 << 9), tag: "li"},
  {mask: (1 << 1), tag: "b"},
  {mask: (1 << 3), tag: "em"},
];

let end_styles = [
  {mask: (1 << 4), tag: "em", para: false},
  {mask: (1 << 2), tag: "b", para: false},
  {mask: (1 << 10), tag: "li", para: true},
  {mask: (1 << 8), tag: "ul", para: true},
  {mask: (1 << 12), tag: "blockquote", para: true},
  {mask: (1 << 6), tag: "h2", para: true},
];

let notchgif = 'data:image/gif;base64,R0lGODlhDAAWAJEAAP/68NK8jv///' +
               'wAAACH5BAUUAAIALAAAAAAMABYAAAIrlI8SmQF83INyNoBtzPhy' +
               'XXHb1ylkZp5dSBqs6KrIq6Xw/FG3V+M9DpkVAAA7';

let next_panel = 1;
let next_docno = 1;
let text_encoder = new TextEncoder("utf-8");
let text_decoder = new TextDecoder("utf-8");

// Token break types.
const NO_BREAK = 0;
const SPACE_BREAK = 1;
const LINE_BREAK = 2;
const SENTENCE_BREAK = 3;
const PARAGRAPH_BREAK = 4;
const SECTION_BREAK = 5;
const CHAPTER_BREAK = 6;

// Document representation from JSON response.
export class Document {
  constructor(data) {
    this.data = data;
    this.docno = next_docno++;
    this.title = data.title;
    this.url = data.url;
    this.key = data.key;
    this.text = text_encoder.encode(data.text);
    this.tokens = data.tokens || [];
    this.frames = data.frames
    this.spans = data.spans
    this.themes = data.themes
    this.SortSpans();
  }

  SortSpans() {
    // Sort spans in nesting order.
    this.spans.sort((a, b) => {
      if (a.begin != b.begin) {
        return a.begin - b.begin;
      } else {
        return b.end - a.end;
      }
    });
  }

  Word(index) {
    let token = this.tokens[index];
    if (token.word) {
      return token.word;
    } else {
      let start = token.start;
      if (start === undefined) start = 0;
      let size = token.size;
      if (size === undefined) size = 1;
      return text_decoder.decode(this.text.slice(start, start + size));
    }
  }

  Phrase(begin, end) {
    if (end == begin + 1) return this.Word(begin);
    let words = [];
    for (let i = begin; i < end; ++i) {
      words.push(this.Word(i));
    }
    return words.join(" ");
  }
}

// Document viewer for displaying spans and frames.
export class DocumentViewer extends Component {
  constructor() {
    super();
    this.document = null;
    this.active_callout = null;
    this.highlighted = null;
    this.labeled = null;
    this.panels = {};
  }

  docno() {
    if (this.document) {
      return this.document.docno;
    } else {
      return 0;
    }
  }

  render(props) {
    // Get document for rendering.
    this.document = props.document;
    if (!this.document) return h(Grid, {class: "docviewer"});

    // Render document text.
    let doc = this.document;
    let spans = doc.spans;
    let nesting = [];
    let styles = [];
    let next = 0;
    let elements = [];
    if (doc.title) {
      let headline = []
      if (doc.url) {
        let props = {href: doc.url, class: "titlelink", target: "_blank"}
        headline.push(h("a", props, doc.title));
      } else {
        headline.push(doc.title);
      }
      if (doc.key) {
        let url = kb_url + doc.key;
        let props = {href: url, class: "topiclink", target: "_blank"}
        headline.push(h("span", {class: "docref"},
                        "(", h("a", props, doc.key), ")"));
      }
      elements.push(h("h1", {class: "title"}, headline));
    } else if (doc.url) {
      elements.push(
        h("div", {class: "title"},
          "url: ",
          h("a", {href: doc.url}, doc.url),
          h("br"),
        )
      );
    }
    for (let index = 0; index < doc.tokens.length; ++index) {
      let token = doc.tokens[index];
      let brk = token.break;

      // Pop elements for end of style.
      if (token.style) {
        for (let ts of end_styles) {
          if (ts.mask & token.style) {
            if (styles.length == 0) break;
            let style = styles.pop();
            let subelements = elements.slice(style.mark);
            elements.length = style.mark;
            elements.push(h(style.tag, null, subelements));
            if (ts.para && brk == PARAGRAPH_BREAK) brk = undefined;
          }
        }
      }

      // Render token break.
      if (index > 0) {
        if (brk === undefined) {
          elements.push(" ");
        } else if (brk >= CHAPTER_BREAK) {
          elements.push(h("hr"));
        } else if (brk >= SECTION_BREAK) {
          elements.push(h("center", null, "***"));
        } else if (brk >= PARAGRAPH_BREAK) {
          elements.push(h("p"));
        } else if (brk >= SENTENCE_BREAK) {
          elements.push(" 	");
        } else if (brk >= SPACE_BREAK) {
          elements.push(" ");
        }
      }

      // Push elements for start of style.
      if (token.style) {
        for (let ts of begin_styles) {
          if (ts.mask & token.style) {
            styles.push({mark: elements.length, tag: ts.tag});
          }
        }
      }

      // Stack spans that begin on this token.
      while (next < spans.length && spans[next].begin == index) {
        let span = spans[next];
        span.mark = elements.length;
        nesting.push(span);
        next++;
      }

      // Render token word.
      let word = doc.Word(index);
      if (word == "``") {
        word = "“";
      } else if (word == "''") {
        word = "”";
      } else if (word == "--") {
        word = "–";
      } else if (word == "...") {
        word = "…";
      }
      elements.push(word);

      // Pop spans that end on this token.
      while (nesting.length > 0 &&
             nesting[nesting.length - 1].end == index + 1) {
        let depth = nesting.length;
        let span = nesting.pop();
        let subelements = elements.slice(span.mark);
        elements.length = span.mark;

        let fidx = span.frame;
        let text = doc.Phrase(span.begin, span.end);
        if (depth > 3) depth = 3;
        let attrs = {
          id: "s" + this.docno() + "-" + fidx,
          frame: fidx,
          class: "b" + depth,
          phrase: text
        };
        elements.push(h("span", attrs, subelements));
      }
    }

    // Terminate remaining styles.
    while (styles.length > 0) {
      let style = styles.pop();
      let subelements = elements.slice(style.mark);
      elements.length = style.mark;
      elements.push(h(style.tag, null, subelements));
    }

    // Add container for theme chips.
    elements.push(h("div", {id: "themes" + this.docno(), class: "themes"}));

    // Render document viewer with text to the left, panels to the right.
    let key = "doc-" + this.docno();
    return (
      h("div", {id: "docview" + this.docno(), class: "docviewer", key},
        h("div", {id: "text" + this.docno(), class: "doctext", key}, elements),
        h("div", {class: "docspacer"}),
        h("div", {id: "panels" + this.docno(), class: "docpanels"})
      )
    );
  }

  componentDidMount() {
    this.Initialize();
  }

  componentDidUpdate() {
    let panels = document.getElementById("panels" + this.docno());
    while (panels.firstChild) panels.removeChild(panels.firstChild);
    this.panels = {}
    this.Initialize();
  }

  Initialize() {
    // Bail out if there is no document.
    if (!this.document) return;

    let docno = this.document.docno;
    for (let i = 0; i < this.document.spans.length; ++i) {
      let fidx = this.document.spans[i].frame;

      // Bind event handlers for spans.
      let span = document.getElementById('s' + docno + "-" + fidx);
      if (span) {
        span.addEventListener('click', this.OpenPanel.bind(this), false);
        span.addEventListener('mouseenter', this.EnterSpan.bind(this), false);
        span.addEventListener('mouseleave', this.LeaveSpan.bind(this), false);
      }

      // Update frame mentions.
      let frame = this.document.frames[fidx];
      for (let s = 0; s < frame.slots.length; s += 2) {
        let value = frame.slots[s + 1];
        if (typeof value == "number") {
          let evoked = this.document.frames[value];
          if (evoked.mentions == null) evoked.mentions = [];
          evoked.mentions.push(fidx);
        }
      }
    }

    // Add chips for themes.
    for (let i = 0; i < this.document.themes.length; ++i) {
      this.AddChip(this.document.themes[i]);
    }
  }

  TypeColor(type) {
    if (type == null) return null;
    let slash = type.indexOf('/', 1);
    if (slash == -1) return null;
    return type_color[type.substring(0, slash)];
  }

  HoverText(frame) {
    let text = '';
    if (frame.id) {
      text += "id: " + frame.id + '\n';
    }
    if (frame.description) {
      text += frame.description + '\n';
    }
    return text;
  }

  FrameName(f)  {
    let name;
    if (typeof f == "number") {
      let frame = this.document.frames[f];
      name = frame.name;
      if (!name) name = frame.id;
      if (!name) name = '#' + f;
    } else {
      name = f;
    }
    return name;
  }

  IsExternal(f) {
    if (typeof f == "number") f = this.document.frames[f];
    if (typeof f == "object") {
      for (let t = 0; t < f.types.length; ++t) {
        let type = f.types[t];
        if (typeof type == "number") {
          let schema = this.document.frames[type];
          if (schema.id == "/w/item") return true;
        }
      }
    }
    return false;
  }

  BuildBox(index, collapsed) {
    let box = document.createElement("div");
    box.className = "boxed";
    box.innerHTML = index;
    box.setAttribute("frame", index);
    box.setAttribute("collapsed", collapsed);
    box.addEventListener('click', this.ClickBox.bind(this), false);
    box.addEventListener('mouseenter', this.EnterBox.bind(this), false);
    box.addEventListener('mouseleave', this.LeaveBox.bind(this), false);
    return box;
  }

  AddTypes(elem, types) {
    if (!types) return;
    for (let t = 0; t < types.length; ++t) {
      let type = types[t];
      let label = document.createElement("span");
      label.className = "type-label";

      let color = null;
      let typename = null;
      if (typeof type == "number") {
        let schema = this.document.frames[type];
        typename = schema.name;
        if (typename) {
          let hover = this.HoverText(schema);
          if (hover.length > 0) {
            label.setAttribute("tooltip", hover);
          }
        } else {
          typename = schema.id;
        }
        color = this.TypeColor(schema.id);
        if (!typename) typename = '(' + t + ')';
      } else {
        typename = type;
        color = this.TypeColor(type);
      }

      if (color) label.style.backgroundColor = color;
      label.appendChild(document.createTextNode(typename));
      elem.appendChild(document.createTextNode(" "));
      elem.appendChild(label);
    }
  }

  BuildAVM(fidx, rendered) {
    let frame = this.document.frames[fidx];
    if (frame == undefined) return document.createTextNode(fidx);
    rendered[fidx] = true;

    let tbl = document.createElement("table");
    tbl.className = "tfs";
    tbl.setAttribute("frame", fidx);

    if (frame.name || frame.id || frame.types.length > 0) {
      let hdr = document.createElement("tr");
      tbl.appendChild(hdr);

      let title = document.createElement("th");
      title.colSpan = 3;
      hdr.appendChild(title);

      if (frame.name || frame.id) {
        let name = document.createTextNode(frame.name ? frame.name : frame.id);
        if (frame.id) {
          if (this.IsExternal(frame)) {
            let a = document.createElement("a");
            a.href = kb_url + frame.id;
            a.target = " _blank";
            a.appendChild(name);
            name = a
          } else {
            let s = document.createElement("span");
            s.appendChild(name);
            name = s;
          }
          name.setAttribute("tooltip", this.HoverText(frame));
        }
        title.appendChild(name);
      }

      this.AddTypes(title, frame.types);
    }

    let slots = frame.slots;
    if (slots) {
      for (let i = 0; i < slots.length; i += 2) {
        let n = slots[i];
        let v = slots[i + 1];

        let row = document.createElement("tr");
        let label = document.createElement("td");
        let box = document.createElement("td");
        let val = document.createElement("td");

        if (typeof n == "number") {
          let span = document.createElement("span");
          let f = this.document.frames[n];
          let role = f.name;
          if (role) {
            let hover = this.HoverText(f);
            if (hover.length > 0) {
              span.setAttribute("tooltip", hover);
            }
          } else {
            role = this.document.frames[n].id;
          }
          if (!role) role = '(' + n + ')';
          span.appendChild(document.createTextNode(role + ':'));
          label.appendChild(span);
        } else {
          label.appendChild(document.createTextNode(n + ':'));
        }

        if (typeof v == "number") {
          let simple = this.document.frames[v].simple == 1;
          box.appendChild(this.BuildBox(v, simple));
          if (rendered[v]) {
            val = null;
          } else {
            if (simple) {
              val.appendChild(this.BuildCollapsedAVM(v));
            } else {
              val.appendChild(this.BuildAVM(v, rendered));
            }
          }
        } else {
          if (this.IsExternal(v)) {
            let a = document.createElement("a");
            a.href = kb_url + v;
            a.target = "_blank";
            a.appendChild(document.createTextNode(v));
            val.appendChild(a);
          } else {
            val.appendChild(document.createTextNode(v));
          }
        }

        row.appendChild(label);
        row.appendChild(box);
        if (val) row.appendChild(val);
        tbl.appendChild(row);
      }
    }

    return tbl;
  }

  BuildCollapsedAVM(fidx) {
    let frame = this.document.frames[fidx];
    let collapsed = document.createElement("span");
    collapsed.className = "tfs-collapsed";
    collapsed.setAttribute("frame", fidx);
    collapsed.appendChild(document.createTextNode(this.FrameName(fidx)));
    return collapsed;
  }

  BuildPanel(phrase, fidx) {
    let frame = this.document.frames[fidx];
    let panel = document.createElement("div");
    panel.className = "panel";
    panel.id = "p" + next_panel++;
    panel.setAttribute("frame", fidx);

    let titlebar = document.createElement("div");
    titlebar.className = "panel-titlebar";
    panel.appendChild(titlebar);

    let title = document.createElement("span");
    title.className = "panel-title";
    if (phrase) {
      title.appendChild(document.createTextNode(phrase));
      titlebar.appendChild(title);
      this.AddTypes(titlebar, frame.types);
    }

    let icon = document.createElement("span");
    icon.className = "panel-icon";
    icon.innerHTML = "&times;";
    icon.setAttribute("panel", panel.id);
    icon.addEventListener('click', this.ClosePanel.bind(this), false);
    titlebar.appendChild(icon);

    let contents = document.createElement("div");
    contents.className = "panel-content"

    if (phrase) {
      let rendered = {};
      let slots = frame.slots;
      if (slots) {
        for (let i = 0; i < slots.length; i += 2) {
          let n = slots[i];
          let v = slots[i + 1];
          if (this.document.frames[n].id == "evokes" ||
              this.document.frames[n].id == "is") {
            let avm = this.BuildAVM(v, rendered);
            contents.appendChild(avm);
          }
        }
      }
    } else {
      let avm = this.BuildAVM(fidx, {});
      contents.appendChild(avm);
    }
    panel.appendChild(contents);
    return panel;
  }

  AddPanel(phrase, fidx) {
    var panel = this.panels[fidx];
    if (panel == undefined) {
      panel = this.BuildPanel(phrase, fidx);
      document.getElementById("panels" + this.docno()).appendChild(panel);
      this.panels[fidx] = panel;
    }
    panel.scrollIntoView();
  }

  OpenPanel(e) {
    e.stopPropagation();
    let span = e.currentTarget;
    let phrase = span.getAttribute("phrase");
    let fidx = parseInt(span.getAttribute("frame"));
    if (phrase) {
      this.AddPanel('"' + phrase + '"', fidx);
    } else {
      this.AddPanel(null, fidx);
    }
  }

  ClosePanel(e) {
    let pid = e.currentTarget.getAttribute("panel");
    let panel =  document.getElementById(pid);
    delete this.panels[panel.getAttribute("frame")];
    document.getElementById("panels" + this.docno()).removeChild(panel);
  }

  BuildChip(fidx) {
    let name = this.FrameName(fidx);
    let chip = document.createElement("span");
    chip.className = "chip";
    chip.id = "t" + fidx;
    chip.setAttribute("frame", fidx);
    chip.appendChild(document.createTextNode(name));

    return chip;
  }

  AddChip(fidx) {
    let chip = this.BuildChip(fidx);
    document.getElementById("themes" + this.docno()).appendChild(chip);
    chip.addEventListener('click', this.OpenPanel.bind(this), false);
    chip.addEventListener('mouseenter', this.EnterChip.bind(this), false);
    chip.addEventListener('mouseleave', this.LeaveChip.bind(this), false);
  }

  AddCallout(span) {
    let callout = document.createElement("span");
    callout.className = "callout";

    let notch = document.createElement("img");
    notch.className = "notch";
    notch.setAttribute("src", notchgif);
    callout.appendChild(notch);

    let bbox = span.getBoundingClientRect();
    callout.style.left = (bbox.right + 15) + "px";
    callout.style.top = ((bbox.top + bbox.bottom) / 2 - 30)  + "px";

    let fidx = parseInt(span.getAttribute("frame"))
    let mention = this.document.frames[fidx];
    let rendered = {};
    let slots = mention.slots;
    if (slots) {
      for (let i = 0; i < slots.length; i += 2) {
        let n = slots[i];
        let v = slots[i + 1];
        if (this.document.frames[n].id == "evokes") {
          let avm = this.BuildAVM(v, rendered);
          callout.appendChild(avm);
        }
      }
    }

    span.appendChild(callout);
    return span;
  }

  RemoveCallout(span) {
    for (let i = 0; i < span.childNodes.length; ++i) {
      let child = span.childNodes[i];
      if (child.className == "callout") {
        span.removeChild(child);
        break;
      }
    }
  }

  GetAVMs(fidx) {
    let matches = null;
    let elements = document.getElementsByClassName("tfs");
    for (let i = 0; i < elements.length; ++i) {
      let e = elements[i];
      let frame = e.getAttribute("frame");
      if (frame == fidx) {
        if (matches == null) matches = [];
        matches.push(e);
      }
    }
    return matches;
  }

  GetBoxes(fidx) {
    let matches = null;
    let elements = document.getElementsByClassName("boxed");
    for (let i = 0; i < elements.length; ++i) {
      let e = elements[i];
      let frame = e.getAttribute("frame");
      if (frame == fidx) {
        if (matches == null) matches = [];
        matches.push(e);
      }
    }
    return matches;
  }

  EvokedFrames(midx) {
    let mention = this.document.frames[midx];
    let evoked = new Set();
    for (let s = 0; s < mention.slots.length; s += 2) {
      let value = mention.slots[s + 1];
      if (typeof value == "number") evoked.add(value);
    }
    return evoked;
  }

  Mentions(evoked) {
    let mentions = new Set();
    for (let fidx of evoked) {
      let frame = this.document.frames[fidx];
      if (frame.mentions) {
        for (let m = 0; m < frame.mentions.length; ++m) {
          mentions.add(frame.mentions[m]);
        }
      }
    }
    return mentions;
  }

  HighlightMentions(mentions) {
    for (let idx of mentions) {
      let span = document.getElementById('s' + this.docno() + '-' + idx);
      span.style.backgroundColor = '#FFFFFF';
      span.style.borderColor = '#FFFFFF';
      span.style.boxShadow = '2px 2px 9px 1px rgba(0,0,0,0.5)';
      this.highlighted.push(span);
    }
  }

  HighlightFrames(evoked) {
    for (let fidx of evoked) {
      let avms = this.GetAVMs(fidx);
      if (avms) {
        for (let i = 0; i < avms.length; ++i) {
          let avm = avms[i];
          avm.style.backgroundColor = '#D0D0D0';
          this.highlighted.push(avm);
        }
      }

      let boxes = this.GetBoxes(fidx);
      if (boxes) {
        for (let i = 0; i < boxes.length; ++i) {
          let box = boxes[i];
          box.style.backgroundColor = '#D0D0D0';
          this.highlighted.push(box);
        }
      }
    }
  }

  LabelMentionedRoles(fidx) {
    let frame = this.document.frames[fidx];
    for (let i = 0; i < frame.slots.length; i += 2) {
      let n = frame.slots[i];
      let v = frame.slots[i + 1];
      if (typeof v == "number") {
        let role = this.FrameName(n);
        let mentions = this.Mentions(new Set([v]));
        for (let idx of mentions) {
          let span = document.getElementById('s' + this.docno() + '-' + idx);
          let label = document.createElement("span");
          label.className = "label";
          label.appendChild(document.createTextNode(role + ':'));
          span.insertBefore(label, span.firstElementChild);
          this.labeled.push(span);
        }
      }
    }
  }

  ClearHighlight() {
    if (this.highlighted) {
      for (let i = 0; i < this.highlighted.length; ++i) {
        this.highlighted[i].removeAttribute("style");
      }
      this.highlighted = null;
    }
    if (this.labeled) {
      for (let i = 0; i < this.labeled.length; ++i) {
        let span = this.labeled[i];
        for (let j = 0; j < span.childNodes.length; ++j) {
          let child = span.childNodes[j];
          if (child.className == "label") span.removeChild(child);
        }
      }
      this.labeled = null;
    }
  }

  EnterSpan(e) {
    if (e.shiftKey) {
      if (this.active_callout) this.RemoveCallout(this.active_callout);
      this.active_callout = this.AddCallout(e.currentTarget);
    } else {
      this.ClearHighlight();
      let span = e.currentTarget;
      let midx = parseInt(span.getAttribute("frame"));

      this.highlighted = [];
      this.labeled = [];
      let evoked = this.EvokedFrames(midx);
      this.HighlightFrames(evoked);
      let corefs = this.Mentions(evoked);
      this.HighlightMentions(corefs);
      for (let fidx of evoked) {
        this.LabelMentionedRoles(fidx);
      }
    }
  }

  LeaveSpan(e) {
    this.RemoveCallout(e.currentTarget);
    this.active_callout = null;
    this.ClearHighlight();
  }

  EnterChip(e) {
    this.ClearHighlight();
    let chip = e.currentTarget;
    let fidx = parseInt(chip.getAttribute("frame"));

    this.highlighted = [];
    this.labeled = [];
    this.HighlightFrames([fidx]);
    this.LabelMentionedRoles(fidx);
  }

  LeaveChip(e) {
    this.ClearHighlight();
  }

  EnterBox(e) {
    if (e.shiftKey) return;
    this.ClearHighlight();
    let box = e.currentTarget;
    let fidx = parseInt(box.getAttribute("frame"));

    this.highlighted = [];
    this.labeled = [];
    let evoked = new Set([fidx]);
    this.HighlightFrames(evoked);
    let corefs = this.Mentions(evoked);
    this.HighlightMentions(corefs);
    this.LabelMentionedRoles(fidx);
  }

  LeaveBox(e) {
    if (e.shiftKey) return;
    this.ClearHighlight();
  }

  ClickBox(e) {
    let box = e.currentTarget;
    let collapsed = box.getAttribute("collapsed") == 1;
    let fidx = parseInt(box.getAttribute("frame"));
    let parent = box.parentElement
    let avm = parent.nextSibling
    if (!avm) return;

    this.ClearHighlight();
    if (collapsed) {
      avm.parentNode.replaceChild(this.BuildAVM(fidx, {}), avm);
      box.setAttribute("collapsed", 0);
    } else {
      avm.parentNode.replaceChild(this.BuildCollapsedAVM(fidx), avm);
      box.setAttribute("collapsed", 1);
    }
  }
}

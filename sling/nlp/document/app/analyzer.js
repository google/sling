import {Component, h, render} from "/common/external/preact.js";
import {Layout, Button, Icon} from "/common/lib/mdl.js";
import {Document, DocumentViewer} from "/common/lib/docview.js";
import {stylesheet} from "/common/lib/util.js";

stylesheet("/doc/analyzer.css");

class DocumentEditor extends Component {
  constructor(props) {
    super(props);
  }

  render(props, state) {
    return (
      h("textarea", {
          id: "text",
          class: "editor",
          oninput: props.oninput,
        },
        props.text)
    );
  }

  componentDidMount() {
    document.getElementById("text").focus();
  }
}

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {document: null, editmode: true};
    this.text = "";
  }

  annotate(e) {
    console.log("annotate", e);
    var self = this;
    let headers = new Headers({
      "Content-Type": "text/lex",
    });
    fetch("/annotate?fmt=cjson", {method: "POST", body: this.text, headers})
      .then(response => {
        if (response.ok) {
          return response.json();
        } else {
          console.log("annotation error", response.status, response.message);
          return null;
        }
      })
      .then(response => {
        self.setState({document: new Document(response), editmode: false});
      });
  }

  edit(e) {
    this.setState({document: null, editmode: true});
  }

  oninput(e) {
    this.text = e.target.value;
  }

  render(props, state) {
    var action, content;
    if (state.editmode) {
      let icon = h(Icon, {icon: "send"});
      action = h(Button,
                 {icon: true, onclick: e => this.annotate(e), accesskey: "g"},
                 icon);
      content = h(DocumentEditor,
                  {text: this.text, oninput: e => this.oninput(e)});
    } else {
      let icon = h(Icon, {icon: "edit"});
      action = h(Button,
                 {icon: true, onclick: e => this.edit(e), accesskey: "g"},
                 icon);
      content = h(DocumentViewer, {document: state.document});
    }

    return (
      h("div", {id: "app"},
        h(Layout, null,
          h(Layout.Header, null,
            h(Layout.HeaderRow, null,
              h(Layout.Title, null, "SLING document analyzer"),
              h(Layout.Spacer),
              action
            ),
          ),
          h(Layout.Drawer, null, h(Layout.Title, null, "Menu")),
          h(Layout.DrawerButton),

          h(Layout.Content, {id: "main"}, content)
        )
      )
    );
  }
}

render(h(App), document.body);

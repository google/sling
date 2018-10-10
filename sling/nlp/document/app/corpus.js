import {Component, h, render} from "/common/external/preact.js";
import {Layout, TextField, Button, Icon} from "/common/lib/mdl.js";
import {Document, DocumentViewer} from "/common/lib/docview.js";
import {stylesheet} from "/common/lib/util.js";

stylesheet("/doc/corpus.css");

class App extends Component {
  constructor(props) {
    super(props);
    this.state = { document: null };
  }

  update(url) {
    var self = this;
    fetch(url)
      .then(response => {
        if (response.ok) {
          return response.json();
        } else {
          console.log("fetch error", response.status, response.message);
          return null;
        }
      })
      .then(response => {
        self.setState({document: new Document(response)});
      });
  }

  search(e) {
    var docid = e.target.value
    if (docid) {
      this.update("/fetch?docid=" + docid + "&fmt=cjson");
    }
  }

  next(e) {
    this.update("/next?fmt=cjson");
  }

  render(props, state) {
    return (
      h("div", {id: "app"},
        h(Layout, null,
          h(Layout.Header, null,
            h(Layout.HeaderRow, null,
              h(Layout.Title, null, "Corpus Browser"),
              h(Layout.Spacer),
              h(TextField, {
                id: "docid",
                placeholder: "Document ID",
                type: "search",
                onsearch: e => this.search(e),
              }),
              h(Button, {icon: true, onclick: e => this.next(e)},
                h(Icon, {icon: "skip_next"})
              ),
            ),
          ),
          h(Layout.Drawer, null, h(Layout.Title, null, "Menu")),
          h(Layout.DrawerButton),

          h(Layout.Content, {id: "main"},
            h(DocumentViewer, {document: state.document})
          )
        )
      )
    );
  }
}

render(h(App), document.body);

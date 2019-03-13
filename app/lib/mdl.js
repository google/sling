// Material Design components for Preact.
// Based on preact-mdl (see https://github.com/developit/preact-mdl/)

import {h, Component} from "/common/external/preact.js";
import {stylesheet} from "/common/lib/util.js";
import "/common/external/material.js";

stylesheet("/common/external/material-icons.css");
stylesheet("/common/external/material.css");

// Extend base with properties.
function extend(base, props) {
  for (let i in props) if (props.hasOwnProperty(i)) base[i] = props[i];
  return base;
}

// Id generator.
let next_uid = 0;

function uid() {
  return ++next_uid;
}

// Base class for Material Design components.
export class MaterialComponent extends Component {
  constructor() {
    super();
    this.component = null;
    this.mdlClasses = null;
    this.js = false;
    this.ripple = false;
  }

  mdlRender(props) {
    return h("div", props, props.children);
  }

  render(props, state) {
    let r = this.mdlRender(props);
    if (!r.attributes) r.attributes = {};
    let name = this.component;

    let c = []
    let js = this.js || this.ripple;
    if (name) c.push(name);
    if (this.mdlClasses) c.push(...this.mdlClasses);
    if (this.ripple) c.push("mdl-js-ripple-effect");
    if (js) c.push(this.component.replace("mdl-", "mdl-js-"));
    if (r.attributes.class) c.push(r.attributes.class);
    r.attributes.class = c.join(" ");
    if (this.nodeName) r.nodeName = this.nodeName;
    if (this.container) r = h("div", {"class": this.container}, r);
    return r;
  }

  componentDidMount() {
    if (this.base && this.base.parentElement) {
      window.componentHandler.upgradeElement(this.base);
    }
  }

  componentWillUnmount() {
    if (this.base && this.base.parentElement) {
      window.componentHandler.downgradeElements(this.base);
    }
  }
}

// Icon component.
export class Icon extends Component {
  render(props) {
    return h("i", {"class": "material-icons"}, props.icon);
  }
}

// Button component.
export class Button extends MaterialComponent {
  constructor() {
    super();
    this.component = 'mdl-button';
    this.nodeName = 'button';
    this.js = true;
    this.ripple = true;
  }

  mdlRender(props) {
    let c = [];
    if (props.class) c.push(props.class);
    if (props.primary) c.push("mdl-button--primary");
    if (props.accent) c.push("mdl-button--accent");
    if (props.colored) c.push("mdl-button--colored");
    if (props.raised) c.push("mdl-button--raised");
    if (props.icon) c.push("mdl-button--icon");
    if (props.fab) c.push("mdl-button--fab");
    if (props.minifab) c.push("mdl-button--mini-fab");
    if (props.disabled) c.push("mdl-button--disabled");
    return h("button",
             Object.assign({class: c.join(" ")}, props),
             props.children);
  }
}

// Tool tip component.
export class Tooltip extends MaterialComponent {
  constructor() {
    super();
    this.component = 'mdl-tooltip';
  }
}

// Text input components.
export class TextField extends MaterialComponent {
  constructor() {
    super();
    this.component = "mdl-textfield";
    this.js = true;
    this.id = uid();
  }

  mdlRender(props) {
    let id = props.id || this.id;
    let p = Object.assign({"class": "mdl-textfield__input",
                           type: "text",
                           id: id}, props);
    return (
      h("div", null,
        h("input", p),
        h("label", {"class": "mdl-textfield__label", "for": id})
      )
    );
  }
}

// Layout components.
export class Layout extends MaterialComponent {
  constructor() {
    super();
    this.component = "mdl-layout";
    this.js = true;
    this.container = "mdl-layout__container";
    this.mdlClasses = ["mdl-layout--fixed-header"];
  }
}

export class LayoutHeader extends MaterialComponent {
  constructor() {
    super();
    this.component = "mdl-layout__header";
    this.nodeName = "header";
  }
}

export class LayoutHeaderRow extends MaterialComponent {
  constructor() {
    super();
    this.component = "mdl-layout__header-row";
  }
}

export class LayoutTitle extends MaterialComponent {
  constructor() {
    super();
    this.component = "mdl-layout-title";
    this.nodeName = "span";
  }
}

export class LayoutSpacer extends MaterialComponent {
  constructor() {
    super();
    this.component = "mdl-layout-spacer";
  }
}

export class LayoutDrawer extends MaterialComponent {
  constructor() {
    super();
    this.component = "mdl-layout__drawer";
  }
}

export class LayoutDrawerButton extends MaterialComponent {
  constructor() {
    super();
    this.component = "mdl-layout__drawer-button";
  }

  mdlRender(props) {
    let p = {role: "button", tabindex: 0, "class": "mdl-layout__drawer-button"};
    return h("div", extend(p, props), h(Icon, {icon: "menu"}));
  }
}

export class LayoutContent extends MaterialComponent {
  constructor() {
    super();
    this.component = "mdl-layout__content";
    this.nodeName = "main";
  }
}

export class LayoutTabBar extends MaterialComponent {
  constructor() {
    super();
    this.component = "mdl-layout__tab-bar";
    this.js = true;
  }
}

export class LayoutTab extends MaterialComponent {
  constructor() {
    super();
    this.component = "mdl-layout__tab";
    this.nodeName = "a";
  }
}

export class LayoutTabPanel extends MaterialComponent {
  constructor() {
    super();
    this.component = "mdl-layout__tab-panel";
  }

  mdlRender(props) {
    return h("section", props,
      h("div", { "class": "mdl-page-content" })
    );
  }
}

extend(Layout, {
  Header: LayoutHeader,
  HeaderRow: LayoutHeaderRow,
  Title: LayoutTitle,
  Spacer: LayoutSpacer,
  Drawer: LayoutDrawer,
  DrawerButton: LayoutDrawerButton,
  Content: LayoutContent,
  TabBar: LayoutTabBar,
  Tab: LayoutTab,
  TabPanel: LayoutTabPanel
});

// Grid components.
export class Grid extends MaterialComponent {
  constructor() {
    super();
    this.component = "mdl-grid";
  }
}

export class Cell extends MaterialComponent {
  constructor() {
    super();
    this.component = "mdl-cell";
  }
}

Grid.Cell = Cell;

// Card components.
export class Card extends MaterialComponent {
  constructor() {
    super();
    this.component = "mdl-card";
  }

  mdlRender(props) {
    let c = [];
    if (props.class) c.push(props.class);
    if (props.shadow) c.push("mdl-shadow--" + props.shadow + "dp");
    return h("div",
             Object.assign({class: c.join(" ")}, props),
             props.children);
  }
}

export class CardTitle extends MaterialComponent {
  constructor() {
    super();
    this.component = "mdl-card__title";
  }
}

export class CardTitleText extends MaterialComponent {
  constructor() {
    super();
    this.component = "mdl-card__title-text";
    this.nodeName = "h2";
  }
}

export class CardMedia extends MaterialComponent {
  constructor() {
    super();
    this.component = "mdl-card__media";
  }
}

export class CardText extends MaterialComponent {
  constructor() {
    super();
    this.component = "mdl-card__supporting-text";
  }
}

export class CardActions extends MaterialComponent {
  constructor() {
    super();
    this.component = "mdl-card__actions";
  }
}

export class CardMenu extends MaterialComponent {
  constructor() {
    super();
    this.component = "mdl-card__menu";
  }
}

extend(Card, {
  Title: CardTitle,
  TitleText: CardTitleText,
  Media: CardMedia,
  Text: CardText,
  Actions: CardActions,
  Menu: CardMenu
});


// Web utility functions.

export function stylesheet(url) {
  if (document.getElementById(url)) return;
  var head  = document.getElementsByTagName('head')[0];
  var link  = document.createElement('link');
  link.id   = url;
  link.rel  = 'stylesheet';
  link.type = 'text/css';
  link.href = url;
  head.appendChild(link);
}


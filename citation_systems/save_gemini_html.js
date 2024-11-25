String.prototype.hashCode = function() {
  var hash = 0,
    i, chr;
  if (this.length === 0) return hash;
  for (i = 0; i < this.length; i++) {
    chr = this.charCodeAt(i);
    hash = ((hash << 5) - hash) + chr;
    hash = hash & 0xffffffff;
    hash = hash >>> 0;
  }
  return hash;
}; // Credit for the code in this function to https://stackoverflow.com/questions/7616461/generate-a-hash-from-string-in-javascript

function citation_number(html_str) {
    let index_of_source = html_str.indexOf('Google Search found similar content, like this');
    let citation_tag = 'entailed citation-end-';
    let index_of_citation = html_str.indexOf(citation_tag);
    let citation_num = html_str.substring(index_of_citation+citation_tag.length, index_of_citation+citation_tag.length+1);
    let prev_citation_num = -1;
    while ((index_of_citation!=-1)&(index_of_source > index_of_citation)) {
        prev_citation_num = citation_num;
        html_str = html_str.substring(index_of_citation+citation_tag.length, html_str.length);
        index_of_citation = html_str.indexOf(citation_tag);
        citation_num = html_str.substring(index_of_citation+citation_tag.length, index_of_citation+citation_tag.length+1);
        citation_num2 = html_str.substring(index_of_citation+citation_tag.length+1, index_of_citation+citation_tag.length+2);
        if (citation_num2 >= '0' && citation_num2 <= '9') {
            citation_num = citation_num.concat(citation_num2)
        }
        index_of_source = html_str.indexOf('Google Search found similar content, like this');
    }
    return prev_citation_num;
};

function save_html() {
  let query_tag = 'class="query-text-line ng-star-inserted">';
  let query_tag_idx = document.documentElement.outerHTML.indexOf(query_tag);
  let query_superstring = document.documentElement.outerHTML.substring(query_tag_idx+query_tag.length, query_tag_idx+500);
  let end_idx = query_superstring.indexOf('\x3C!');
  let query = query_superstring.substring(0, end_idx);
  let citation_n = citation_number(document.documentElement.outerHTML);

  let blob = new Blob([document.documentElement.outerHTML], {type: 'text/plain'});
  let url = URL.createObjectURL(blob);
  let downloadLink = document.createElement('a');
  downloadLink.download = query.hashCode().toString().concat('_', citation_n).concat('.txt');
  downloadLink.href = url;
  downloadLink.click();
};

save_html();
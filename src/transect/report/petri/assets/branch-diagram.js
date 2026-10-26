'use strict';
(() => {
  function mount(configNode) {
  if(configNode.dataset.petriBranchesMounted)return;
  configNode.dataset.petriBranchesMounted='yes';
  const config=JSON.parse(configNode.textContent);
  const prefix=config.prefix || '';
  const containerId=config.container_id || 'phase-cards-0';
  const data=config.diagram;
  if (!data) return;
  const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!=null)n.textContent=text;if(cls)n.className=cls;return n;};
  const svg=(tag,attrs={},text)=>{const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(text!=null)n.textContent=text;return n;};
  const existing=document.getElementById(config.branch_container_id || prefix+'branch-overview');
  const section=existing || el('section',null,'section branch-overview');section.id=prefix+'branch-overview';section.replaceChildren();
  section.append(el('h3','Target conversation branches'));
  const cards=document.getElementById(containerId);
  if (!cards) return;
  if(!existing)(cards.closest('.section') || cards).before(section);
  const evidence=data.evidence || {};
  if(evidence.title) {
    const verdict=el('p',null,'branch-verdict');
    verdict.append(el('strong',evidence.title));
    if(evidence.label)verdict.append(' · '+evidence.label);
    if(evidence.detail)verdict.append(' · '+evidence.detail);
    if(evidence.href){const a=el('a','Evidence details');a.href=evidence.href;verdict.append(' · ',a);}
    section.append(verdict);
    if(evidence.description)section.append(el('p',evidence.description,'meta'));
    if(evidence.status==='located') {
      const a=el('a','Jump to '+(evidence.badge || 'located evidence'));a.href=`#${prefix}turn-${evidence.turns[0]}`;section.append(a);
    }
    if(evidence.quote) {
      const details=el('details',null,'branch-flags');details.append(el('summary','Saved evidence'));
      details.append(el('blockquote',evidence.quote,'branch-quote'));
      for(const field of evidence.fields || [])details.append(el('p',field.label+': '+(field.values.join(', ') || 'None recorded')));
      if(evidence.status==='ambiguous')for(const number of evidence.turns){const a=el('a',`Candidate target call #${number}`);a.href=`#${prefix}turn-${number}`;details.append(a,' ');}
      section.append(details);
    }
  }
  if(data.status!=='ok'){section.append(el('p',data.reason,'meta'));return;}
  const rollbacks=data.branches.filter(b=>b.rollback?.kind==='rollback_conversation').length;
  section.append(el('p',`${data.branches.length} recorded branches · ${rollbacks} rollback points · ${data.target_calls.toLocaleString()} target calls. Lines follow the saved Inspect conversation tree. Click a phase segment to open its first target turn; a fork circle opens its auditor rollback when available. Horizontal distance shows conversation progress, not elapsed time.`,'meta'));
  if(data.unresolved)section.append(el('p',`${data.unresolved} branch connection(s) lack a retained parent anchor and are shown disconnected.`,'meta'));
  const toolbar=el('div',null,'filter-bar branch-controls');section.append(toolbar);
  const viewport=el('div',null,'branch-viewport');viewport.tabIndex=0;viewport.setAttribute('aria-label','Scrollable target conversation branch diagram');section.append(viewport);
  const hint=el('p',null,'meta');
  hint.append(el('span','','branch-swatch branch-swatch-phase'),'Phase segments · ');
  if(evidence.title)hint.append(el('span','','branch-swatch branch-swatch-evidence'),(evidence.badge || 'Located evidence')+' · ');
  hint.append('Circle: branch start / rollback. Scroll to see more branches; zoom in for shorter phase labels.');
  if(data.final_path_status==='ok')hint.prepend('Solid colour: surviving path · Muted: discarded continuation · ');
  section.append(hint);
  let unit=32;
  const left=75, top=32, rowHeight=54;
  const phase=card=>card.startsWith('phase-')?'Phase '+card.split('-').at(-1):'Unclassified';
  const rows=new Map(data.branches.map((b,i)=>[b.id,i]));
  const maxEnd=Math.max(1,...data.branches.map(b=>b.end));
  function link(parent,href,label){
    const a=svg('a',{href,tabindex:'0','aria-label':label});a.append(svg('title',{},label));parent.append(a);return a;
  }
  function render(){
    const width=Math.max(left+maxEnd*unit+30,viewport.clientWidth-2);
    const chart=svg('svg',{width,height:top+data.branches.length*rowHeight+16,role:'group','aria-label':'Recorded target branches and clickable phase sections'});
    chart.append(svg('text',{x:left,y:15,class:'branch-axis'},'Target conversation progress →'));
    for(const [i,b] of data.branches.entries()){
      const x=left+b.start*unit,y=top+i*rowHeight;
      if(b.parent){
        const py=top+rows.get(b.parent)*rowHeight;
        chart.append(svg('path',{d:`M ${x} ${py} L ${x} ${y-9} Q ${x} ${y} ${x+9} ${y}`,class:'branch-connector'+(b.surviving?' branch-surviving-connector':''),'data-parent':b.parent,'data-child':b.id}));
      }
    }
    for(const [i,b] of data.branches.entries()){
      const x=left+b.start*unit,y=top+i*rowHeight;
      const row=svg('g',{'data-branch':b.number});chart.append(row);
      const startTurn=b.rollback?.turn || b.sections[0]?.turns[0];
      const label=`Branch ${b.number}${b.rollback?.destination?' · rollback to '+b.rollback.destination:''}${b.unresolved?' · parent anchor unavailable':''} · ${b.target_calls} target calls`;
      const rowLabel=startTurn?link(row,`#${prefix}turn-${startTurn}`,label):row;
      rowLabel.append(svg('text',{x:8,y:y+4,class:'branch-row-label'},`B${b.number}`));
      row.append(svg('line',{x1:x,y1:y,x2:left+b.end*unit+4,y2:y,class:'branch-track'}));
      if(b.surviving_end!=null)row.append(svg('line',{x1:x,y1:y,x2:left+b.surviving_end*unit,y2:y,class:'branch-surviving-track'}));
      const fork=startTurn?link(row,`#${prefix}turn-${startTurn}`,label):row;
      fork.append(svg('circle',{cx:x,cy:y,r:5,class:'branch-fork'}));
      for(const s of b.sections){
        const sx=left+s.start*unit,sw=Math.max(5,(s.end-s.start)*unit);
        const title=`Branch ${b.number} · ${phase(s.card)}${s.surviving==null?'':s.surviving?' · surviving path':' · discarded continuation'} · ${s.turns.length} target call${s.turns.length===1?'':'s'} · opens call #${s.turns[0]}${s.flag?' · '+(evidence.badge || 'Located evidence'):''}`;
        const a=link(row,`#${prefix}turn-${s.turns[0]}`,title);a.classList.add('branch-segment');a.dataset.card=s.card;
        if(s.flag)a.classList.add('branch-evidence-link');
        a.classList.toggle('branch-discarded',s.surviving===false);
        a.classList.toggle('branch-surviving',s.surviving===true);
        a.append(svg('rect',{x:sx,y:y-10,width:sw,height:20,rx:3,class:s.flag?'branch-block branch-flagged':'branch-block'}));
        const short=s.card.startsWith('phase-')?'P'+s.card.split('-').at(-1):'?';
        if(sw>24)a.append(svg('text',{x:sx+sw/2,y:y+4,'text-anchor':'middle',class:'branch-phase-label'},sw>92?phase(s.card):short));
      }
      if(!b.sections.length)row.append(svg('text',{x:x+12,y:y+23,class:'branch-axis'},'No target call recorded'));
    }
    viewport.replaceChildren(chart);
  }
  function button(title,fn){const b=el('button',title);b.type='button';b.onclick=fn;toolbar.append(b);return b;}
  button('Zoom out',()=>{unit=Math.max(4,unit/1.4);render();});
  button('Zoom in',()=>{unit=Math.min(100,unit*1.4);render();});
  function fit(){unit=Math.max(4,Math.min(32,(viewport.clientWidth-left-35)/maxEnd));render();}
  button('Fit width',fit);
  const expand=button('Show full height',()=>{const open=viewport.classList.toggle('branch-expanded');expand.textContent=open?'Limit height':'Show full height';});
  if(evidence.status==='located')button('Show evidence in diagram',()=>{
    const mark=viewport.querySelector('.branch-evidence-link');
    if(mark){const box=mark.getBBox();viewport.scrollTo({left:Math.max(0,box.x-100),top:Math.max(0,box.y-100),behavior:'smooth'});mark.focus({preventScroll:true});}
  });
  section.addEventListener('click',event=>{
    const a=event.target.closest(`a[href^="#${prefix}turn-"]`);
    if(a && location.hash===a.getAttribute('href'))window.dispatchEvent(new HashChangeEvent('hashchange'));
  });
  fit();
  }
  for(const node of document.querySelectorAll('script[data-petri-config], #phase-transcript-config'))mount(node);
})();

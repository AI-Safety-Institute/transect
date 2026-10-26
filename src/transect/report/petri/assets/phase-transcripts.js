'use strict';
(() => {
  function mount(configNode) {
  if (configNode.dataset.petriMounted) return;
  configNode.dataset.petriMounted='yes';
  const config = JSON.parse(configNode.textContent);
  const root=configNode.closest('.transcript') || document;
  const prefix=config.prefix || '';
  const containerId=config.container_id || 'phase-cards-0';
  const auditorContainerId='auditor-'+containerId;
  const localId=name=>prefix+name;
  const localHash=name=>'#'+localId(name);
  const turnPattern=new RegExp('^#'+prefix+'turn-(\\d+)$');
  const path=config.path;
  const focused=path?.status==='ok';
  const groups=focused ? path.groups : config.groups;
  const el = (tag, text, cls) => {
    const node = document.createElement(tag);
    if (text != null) node.textContent = text;
    if (cls) node.className = cls;
    return node;
  };
  function disclosure(title, render, cls='full-detail') {
    const node = el('details', null, cls);
    node.append(el('summary', title));
    node.addEventListener('toggle', () => { if (node.open) render(node); });
    return node;
  }
  function once(title, render, cls) {
    let loaded = false;
    return disclosure(title, node => { if (!loaded) { loaded=true; render(node); } }, cls);
  }
  let matchingTurns = null;
  let pending;
  async function payload() {
    if(config.payload_id) {
      if(!pending) pending=Promise.resolve().then(()=>JSON.parse(document.getElementById(config.payload_id).textContent));
      return pending;
    }
    if (!pending) pending = (async () => {
      if ('DecompressionStream' in window) {
        try {
          const response = await fetch(config.url+'.gz', {cache:'no-store'});
          if (response.ok) return await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).json();
        } catch (_) { /* Plain JSON also works behind proxies without gzip sidecars. */ }
      }
      const response = await fetch(config.url, {cache:'no-store'});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })().then(data => {
      if (data.turns.length!==config.total) throw new Error('Transcript population does not match this report.');
      return data;
    }).catch(error => { pending=null; throw error; });
    return pending;
  }
  async function loadInto(node, render) {
    if (node.dataset.loaded) return;
    if (node._loading) return node._loading;
    node.querySelector('.full-load-status')?.remove();
    const status=el('p','Loading full transcript…','meta full-load-status');
    status.setAttribute('role','status'); node.append(status);
    node._loading=(async () => {
      try {
        const data=await payload();
        await render(data);
        node.dataset.loaded='yes'; status.remove();
      } catch (error) {
        status.textContent=`Could not load the transcript (${error.message}). `;
        const retry=el('button','Retry');retry.type='button';
        retry.onclick=()=>loadInto(node,render);status.append(retry);
      } finally { node._loading=null; }
    })();
    return node._loading;
  }
  function message(data, record, isInput=false) {
    const box=el('div',null,'full-message');
    if (isInput) {
      const labels={system:'System instructions',user:'User message supplied to the model',tool:'Tool result supplied to the model',assistant:record.generated?'Prior generated assistant turn':'Assistant context / prefill supplied to the model'};
      box.append(el('p',labels[record.role] || record.role,'meta'));
    }
    if (record.text) box.append(el('div',record.text,'full-prose'));
    if (record.reasoning) box.append(once('Recorded reasoning / summary',node=>node.append(el('div',record.reasoning,'full-prose'))));
    for (const call of record.tools) {
      const tool=el('div',null,'full-tool');tool.append(el('strong',`Tool call: ${call.name}`));
      tool.append(once('Full arguments',node=>node.append(el('pre',JSON.stringify(call.arguments,null,2)))));
      if (call.result != null) tool.append(once('Full tool result',node=>node.append(el('pre',call.result))));
      if (call.error != null) tool.append(el('pre',JSON.stringify(call.error,null,2)));
      box.append(tool);
    }
    if (!record.text && !record.reasoning && !record.tools.length) box.append(el('p','No readable text or tool call recorded.','meta'));
    return box;
  }
  function turnNode(data, turn) {
    const label=turn.role==='target'
      ? `Target · ${config.target_label || turn.model || data.target_model || config.target_model || "Target model"} · attempt ${turn.attempt} · turn ${turn.lane_turn+1}`
      : `Auditor · turn ${turn.role_turn-1}`;
    const node=once(label,body=>{
      if (focused && turn.role==='target' && path.auditor_context[turn.number]) {
        const context=el('a',`Auditor context · call #${path.auditor_context[turn.number]}`);
        context.href=`#${prefix}turn-${path.auditor_context[turn.number]}`;body.append(context);
      }
      body.append(message(data,data.messages[turn.output]));
      if (turn.inputs.length) body.append(once(`Full context seen by ${turn.role} (${turn.inputs.length} messages)`,context=>{
        for (const index of turn.inputs) context.append(message(data,data.messages[index],true));
      },'full-detail full-context'));
    },`full-turn full-${turn.role}`);
    node.id=localId(`turn-${turn.number}`);
    node.classList.toggle('full-search-match', matchingTurns?.has(turn.number) || false);
    const summary=node.querySelector('summary');
    const anchor=el('a',`#${turn.number}`);anchor.href=`#${prefix}turn-${turn.number}`;
    summary.append(' ',anchor);
    if (turn.citations.length) summary.append(el('span',' · '+turn.citations.join(', '),'meta'));
    return node;
  }
  let container=document.getElementById(containerId);
  if (!container) {
    const section=el('section',null,'section');section.append(el('h3','Phase cards'));
    container=el('div');container.id=containerId;section.append(container);
    const footer=root.querySelector('.report-footer');
    if (footer) footer.before(section); else document.body.append(section);
  }
  const categoryNode=root.querySelector('script[data-petri-categories]');
  const categories=config.categories || (categoryNode ? JSON.parse(categoryNode.textContent) : null);
  const awareness=categories?.turns || {};
  const evidence=config.diagram?.evidence || {};
  const quoteAvailable=['located','ambiguous'].includes(evidence.status);
  const quoteTurns=new Set(quoteAvailable ? evidence.turns : []);
  const evidenceTitle=evidence.title || 'Evidence';
  const evidenceBadge=evidence.badge || 'Located evidence';
  function flags(numbers) {
    const box=el('span',null,'full-path-flags');
    const quoted=numbers.filter(n=>quoteTurns.has(n));
    if (quoted.length) box.append(el('span',evidence.status==='located' ? evidenceBadge : `${quoted.length} ${evidenceTitle} quote candidates`,'full-evidence-flag'));
    for (const key of categories?.flagged || []) {
      const count=numbers.filter(n=>awareness[n]===key).length;
      if (count) box.append(el('span',`${count} ${categories.labels.find(([value])=>value===key)?.[1] || key}`,'full-awareness-flag'));
    }
    return box;
  }
  if (focused) {
    // The existing auditor judgments and their filters remain intact and scoped
    // separately. Sorting those assessments cannot rearrange the target path.
    const old=container;
    old.id=auditorContainerId;
    const reference=el('details',null,'full-original');reference.id=localId('auditor-phase-assessments');
    reference.append(el('summary','Auditor phase assessments and filters'));
    reference.append(el('p','These original judgments describe the auditor. A phase may span both surviving and discarded target calls; its narrative and confidence apply to the complete original phase.','meta'));
    const bar=root.querySelector(`.filter-bar[data-scope="${containerId}"]`);
    if (bar) {
      bar.dataset.scope=old.id;
      for (const node of [bar,...bar.querySelectorAll('*')]) {
        for (const attr of [...node.attributes]) {
          if (['id','for','onchange','onclick'].includes(attr.name)) node.setAttribute(attr.name,attr.value.replaceAll(containerId,old.id));
        }
      }
      reference.append(bar);
    }
    const count=document.getElementById(containerId+'-count');
    if (count) {count.id=old.id+'-count';reference.append(count);}
    const empty=document.getElementById(containerId+'-empty');
    if (empty) empty.id=old.id+'-empty';
    container=el('div',null,'target-phase-cards');container.id=containerId;
    old.before(container);reference.append(old);container.before(reference);
    reference.hidden=!old.querySelector('.event');
    const heading=container.closest('.section')?.querySelector('h3');
    if (heading) heading.textContent='Target conversation';
  }
  const intro=el('p',focused
    ? `${config.target_label || config.target_model || "Target"} · surviving conversation · ${path.surviving_turns.length} target calls. ${path.discarded_turns.length} discarded target calls remain folded at their divergence points and still count in the assessments. Open a phase and a target turn for complete text, reasoning, tools, results and model context. Auditor calls are folded separately; no text is truncated.`
    : `Full retained conversation: ${config.total.toLocaleString()} model calls. Expand a phase’s transcript, then a turn to read its complete text, tool calls, results and model context. Phase labels describe the auditor. No display length limits are applied.${path?.reason ? ' '+path.reason : ''}`,
    'meta explainer full-transcript-intro');
  container.before(intro);
  if(config.source_note)container.before(el('p',config.source_note,'meta'));
  const hasRollbacks=!focused && config.groups.some(group=>group.segments.some(segment=>segment.rollbacks.length));
  if (hasRollbacks) container.before(el('p','Continuation numbers follow recorded call order. The branch overview shows the saved Inspect ancestry. Tools within one auditor turn may span the rollback, so phase grouping alone does not reconstruct target context.','meta full-branch-explainer'));
  const panels=new Map();
  const turnGroups=new Map();
  function populate(group,node) {
    return loadInto(node,data=>{
      const toolbar=el('div',null,'filter-bar');
      for (const [title,open] of [[focused?'Expand target turns':'Expand all turns',true],[focused?'Collapse target turns':'Collapse all turns',false]]) {
        const button=el('button',title);button.type='button';
        button.onclick=()=>node.querySelectorAll(focused?'.full-target':'.full-turn').forEach(turn=>{turn.open=open;});
        toolbar.append(button);
      }
      node.append(toolbar);
      if (focused) {
        if (group.auditor_turns.length) {
          const audit=el('details',null,'full-detail full-auditor-context');
          audit.append(el('summary',`Auditor context · ${group.auditor_turns.length} calls`));
          audit.append(el('p','Recorded auditor calls associated with this section. One auditor call can control multiple attempts; its complete output is retained once.','meta'));
          for (const number of group.auditor_turns) audit.append(turnNode(data,data.turns[number-1]));
          node.append(audit);
        }
        for (const number of group.target_turns) node.append(turnNode(data,data.turns[number-1]));
        return;
      }
      for (const segment of group.segments) {
        const section=el('div',null,hasRollbacks ? 'full-continuation' : '');
        if (hasRollbacks) section.append(el('p',`${segment.continuation===1?'Initial continuation':'Retry continuation'} ${segment.continuation} · recorded turns ${segment.turns[0]}–${segment.turns.at(-1)}`,'meta full-continuation-label'));
        for (const number of segment.turns) section.append(turnNode(data,data.turns[number-1]));
        node.append(section);
        for (const event of segment.rollbacks) {
          const marker=el('div',null,'full-rollback');
          marker.dataset.status=event.status;
          const destination=event.destination==null?'unrecorded destination':String(event.destination);
          const title=event.status==='confirmed'?`↶ Rollback to ${destination}`
            :event.status==='failed'?`Rollback failed · ${destination}`:`Rollback unconfirmed · ${destination}`;
          marker.append(el('strong',title));
          const link=el('a',`auditor call #${event.turn}, tool ${event.tool_index+1}`);link.href=`#${prefix}turn-${event.turn}`;
          marker.append(' · ',link);
          marker.append(el('p',event.status==='confirmed'
            ?`Continuation ${event.next_continuation} follows this recorded rollback. ${config.diagram?.status==='ok'?'See the branch overview for saved ancestry.':'Exact parent turn not mapped.'} Earlier text is retained.`
            :'No confirmed continuation change. Open the auditor call for the recorded arguments and result.','meta'));
          node.append(marker);
        }
      }
    });
  }
  const alternativePanels=[];
  if (focused) {
    const groupMap=new Map(groups.map(g=>[g.card,g]));
    function cardFor(id) {
      const group=groupMap.get(id);
      const phase=group.phase.startsWith('phase-') ? 'Phase '+group.phase.split('-').at(-1) : 'Unclassified phase';
      const title=group.target_turns.length
        ? `${phase} · ${group.target_turns.length} target call${group.target_turns.length===1?'':'s'} · ${group.surviving?'surviving path':'earlier attempt · B'+group.branch}`
        : 'Closing auditor context';
      const card=disclosure(title,node=>populate(group,node),'event full-target-phase');card.id=group.card;
      card.dataset.surviving=String(group.surviving);card.dataset.phase=group.phase;
      card.querySelector('summary').append(flags(group.target_turns));
      const native=document.getElementById(group.phase);
      if (native) {
        const summary=card.querySelector(':scope > summary');
        const swatch=native.querySelector(':scope > summary > .swatch');
        if (swatch) summary.prepend(swatch.cloneNode(true),' ');
        const classification=native.querySelector(':scope > summary > .class-box');
        if (classification) {
          const label=classification.cloneNode(true);
          label.title='Saved judgment of the complete original auditor phase, including surviving and discarded calls. '+(label.title || '');
          const line=el('span',null,'full-phase-classification');
          line.append(el('span','Auditor phase: ','meta'),label);
          summary.append(line);
        }
        const note=el('p',null,'meta');const link=el('a',`${phase} · auditor assessment${native.dataset.label?' · '+native.dataset.label.replaceAll('_',' '):''}`);
        link.href='#'+group.phase;note.append(link);card.append(note);
      }
      panels.set(group.card,card);
      for (const number of group.turns) turnGroups.set(number,group);
      return card;
    }
    for (const entry of path.entries) {
      if (entry.kind==='phase') {container.append(cardFor(entry.card));continue;}
      const box=el('details',null,'full-detail full-alternatives');
      const where=entry.after_turn ? `after target call #${entry.after_turn}` : 'before the first surviving target call';
      const title=el('summary',`Earlier attempts · ${entry.attempts} attempt${entry.attempts===1?'':'s'} · ${entry.turns.length} target call${entry.turns.length===1?'':'s'} · ${where}`);
      title.append(flags(entry.turns));box.append(title);
      box.append(el('p','These continuations were discarded. Their actions remain part of the report’s judgments. Shared surviving history appears once in the main conversation; B labels match the branch diagram.','meta'));
      for (const branch of entry.branches) {
        const note=el('p',`B${branch.branch}${branch.parent?' · branches from B'+branch.parent:' · original continuation'}`,'meta full-attempt-ancestry');
        if (branch.rollback) {
          const link=el('a',`Auditor rollback · #${branch.rollback.turn}`);link.href=`#${prefix}turn-${branch.rollback.turn}`;note.append(' · ',link);
        }
        box.append(note);
        for (const id of entry.cards.filter(id=>groupMap.get(id).branch===branch.branch)) box.append(cardFor(id));
      }
      alternativePanels.push({node:box,cards:entry.cards});container.append(box);
    }
  }
  if (!focused) for (const group of config.groups) {
    const count=`${group.roles.auditor || 0} auditor, ${group.roles.target || 0} target turns`;
    let panel;
    if (group.card.endsWith('unclassified-transcript')) {
      panel=disclosure(`Turns without phase labels · ${count}`,node=>populate(group,node),'full-unclassified');
      panel.id=group.card;
      panel.append(el('p','These calls have no recorded auditor phase label, or precede the first auditor call. The transcript search below also includes these calls; native phase-label filters do not apply to them.','meta'));
      container.after(panel);
    } else {
      const card=document.getElementById(group.card);
      if (hasRollbacks) {
        const numbers=[...new Set(group.segments.map(segment=>segment.continuation))];
        const events=group.segments.flatMap(segment=>segment.rollbacks);
        const confirmed=events.filter(event=>event.status==='confirmed').length;
        const uncertain=events.length-confirmed;
        const range=numbers.length===1?`${numbers[0]}`:`${numbers[0]}–${numbers.at(-1)}`;
        const badge=el('span',` · Continuation ${range}${confirmed?` · ${confirmed} rollback${confirmed===1?'':'s'}`:''}${uncertain?` · ${uncertain} failed / unconfirmed`:''}`,'meta full-branch-badge');
        card.querySelector(':scope > summary').append(badge);
        card.classList.add('full-branch-phase');
        if (confirmed) card.classList.add('full-rollback-phase');
      }
      panel=disclosure(`Full transcript · ${count}`,node=>populate(group,node),'full-conversation');
      card.append(panel);
    }
    panels.set(group.card,panel);
    for (const number of group.turns) turnGroups.set(number,group);
  }
  // These controls filter retained output without mounting the whole transcript.
  // Category and quote constraints intersect on an individual call, not a phase.
  const normalize=text=>String(text ?? '').replace(/\s+/g,' ').trim().toLowerCase();
  const filterBox=el('div',null,'full-search-controls');
  filterBox.id=localId('transcript-search');
  filterBox.append(el('h4','Search phase transcripts'));
  const fields=el('div',null,'filter-bar full-search-fields');
  const search=el('input');search.type='search';search.id=localId('transcript-query');
  search.placeholder='Search recorded text, reasoning or tools';
  const searchLabel=el('label','Recorded output ');searchLabel.htmlFor=search.id;searchLabel.append(search);
  const category=el('select');category.id=localId('transcript-awareness');
  for (const [key,title] of [['','All categories'],...(categories?.labels || [])]) {
    const option=el('option',title);option.value=key;category.append(option);
  }
  category.disabled=!categories;
  const categoryLabel=el('label',(categories?.title || 'Category')+' ');categoryLabel.htmlFor=category.id;categoryLabel.append(category);categoryLabel.hidden=!categories;
  const quote=el('input');quote.type='checkbox';quote.id=localId('transcript-worst-quote');quote.disabled=!quoteAvailable;
  const quoteLabel=el('label');quoteLabel.htmlFor=quote.id;quoteLabel.append(quote,` ${evidenceTitle} quote matches`);quoteLabel.hidden=!evidence.quote;
  const reset=el('button','Clear transcript filters');reset.type='button';
  fields.append(searchLabel,categoryLabel,quoteLabel,reset);filterBox.append(fields);
  filterBox.append(el('p','Search covers recorded outputs, reasoning, tool arguments and results, excluding repeated model context. Quote and category filters select target calls; all selected filters must match the same call. Matching phases retain their full conversation.','meta'));
  if (evidence.quote) {
    const details=el('details',null,'full-detail');details.append(el('summary',evidenceTitle+' direct quote'));
    details.append(el('blockquote',evidence.quote,'full-prose'));
    const note=evidence.status==='ambiguous' ? 'This quote occurs in multiple target calls; all candidates are included.'
      :evidence.status==='initial_vote' ? 'The verifier overturned the original verdict. This original-vote quote is not used as final evidence.'
      :quoteAvailable ? 'Matches use the saved location in generated target output, excluding replayed context.'
      :'The quote could not be located in retained target output; the quote filter is unavailable.';
    details.append(el('p',note,'meta'));filterBox.append(details);
  }
  const searchStatus=el('p',null,'meta');searchStatus.id=localId('transcript-search-status');searchStatus.setAttribute('role','status');
  const results=el('details',null,'full-detail full-search-results');results.hidden=true;
  const resultsTitle=el('summary');const resultLinks=el('div',null,'full-search-links');results.append(resultsTitle,resultLinks);
  filterBox.append(searchStatus,results);container.before(filterBox);
  let revision=0, timer, indexedText;
  function showMatches(matches) {
    matchingTurns=matches;
    let visible=0, matched=0;
    for (const group of groups) {
      const card=document.getElementById(group.card);
      const numbers=matches ? group.turns.filter(n=>matches.has(n)) : group.turns;
      card.classList.toggle('full-search-hidden', !numbers.length);
      card.querySelector(':scope > summary > .full-match-badge')?.remove();
      if (matches && numbers.length) card.querySelector(':scope > summary')?.append(el('span',` · ${numbers.length} matching call${numbers.length===1?'':'s'}`,'meta full-match-badge'));
      if (numbers.length) {matched++;if(card.style.display!=='none')visible++;}
      for (const turn of card.querySelectorAll('.full-turn')) turn.classList.toggle('full-search-match', matches?.has(Number(turn.id.slice(prefix.length+5))) || false);
    }
    for (const alternative of alternativePanels) {
      const shown=alternative.cards.some(id=>!document.getElementById(id).classList.contains('full-search-hidden'));
      alternative.node.classList.toggle('full-search-hidden',!shown);
      if (matches && shown) alternative.node.open=true;
    }
    searchStatus.textContent=focused
      ? (matches ? `${matches.size} matching calls in ${matched} sections, including earlier attempts.` : 'Search includes both the surviving conversation and all earlier attempts.')
      : (matches ? `${matches.size} matching calls in ${matched} phase groups; ${visible} groups visible with the native phase filters.` : 'All transcript calls included. Native phase filters still apply.');
    resultLinks.replaceChildren();results.hidden=!matches || !matches.size;
    resultsTitle.textContent=`Matching calls (${matches?.size || 0})`;
    if (matches) for (const number of [...matches].sort((a,b)=>a-b)) {
      const link=el('a',`#${number}${awareness[number] ? ' · '+awareness[number].replaceAll('_',' ') : ''}`);link.href=`#${prefix}turn-${number}`;resultLinks.append(link);
    }
  }
  async function applyFilters() {
    const current=++revision;
    const query=normalize(search.value), wanted=category.value, onlyQuote=quote.checked;
    const active=query || wanted || onlyQuote;
    if (!active) {showMatches(null);return;}
    try {
      if (query && !indexedText) {
        searchStatus.textContent='Loading recorded output for search…';
        const data=await payload();
        if (current!==revision) return;
        indexedText=new Map(data.turns.map(turn=>{
          const record=data.messages[turn.output];
          const parts=[record.text,record.reasoning,...record.tools.flatMap(tool=>[tool.name,JSON.stringify(tool.arguments),tool.result,JSON.stringify(tool.error)])];
          return [turn.number,normalize(parts.filter(v=>v!=null).join('\n'))];
        }));
      }
      if (current!==revision) return;
      showMatches(new Set([...turnGroups.keys()].filter(number=>(!wanted || awareness[number]===wanted) && (!onlyQuote || quoteTurns.has(number)) && (!query || indexedText.get(number)?.includes(query)))));
    } catch (error) {
      if (current!==revision) return;
      showMatches(null);searchStatus.textContent=`Search could not load the transcript (${error.message}). `;
      const retry=el('button','Retry search');retry.type='button';retry.onclick=applyFilters;searchStatus.append(retry);
    }
  }
  function clearFilters() {
    clearTimeout(timer);revision++;search.value='';category.value='';quote.checked=false;showMatches(null);
  }
  reset.onclick=clearFilters;
  search.addEventListener('input',()=>{revision++;clearTimeout(timer);timer=setTimeout(applyFilters,180);});
  category.addEventListener('change',applyFilters);quote.addEventListener('change',applyFilters);
  root.querySelector(`.filter-bar[data-scope="${containerId}"]`)?.addEventListener('change',()=>setTimeout(()=>showMatches(matchingTurns),0));
  showMatches(null);
  async function jump(number) {
    const group=turnGroups.get(number);
    if (!group) return;
    const card=document.getElementById(group.card);
    if (card.style.display==='none' && window._transectClearCardFilters) window._transectClearCardFilters(containerId);
    if (card.classList.contains('full-search-hidden')) clearFilters();
    card.open=true;
    const panel=panels.get(group.card);panel.open=true;
    await populate(group,panel);
    const turn=document.getElementById(localId(`turn-${number}`));
    if (turn) {
      for (let parent=turn;parent;parent=parent.parentElement) if (parent.tagName==='DETAILS') parent.open=true;
      // Disclosure rendering is lazy; wait for layout before positioning a
      // formerly hidden alternative or its folded auditor context.
      await new Promise(resolve=>requestAnimationFrame(resolve));
      turn.scrollIntoView({block:'start'});
    }
  }
  const original=config.original_evaluation ? disclosure('Original evaluation score and cited evidence',node=>loadInto(node,data=>{
    node.append(el('p',`Original score: ${data.score==null?'Not scored':data.score+'/10'} · original judge: ${data.judge_model || 'not recorded'}`,'meta'));
    node.append(el('div',data.score_explanation || 'No original explanation recorded.','full-prose'));
    for (const reference of data.references) {
      if (!reference.match) {node.append(el('p',`${reference.cite}: referenced message was not retained.`,'meta'));continue;}
      node.append(once(`${reference.cite} · ${reference.match.location==='input'?'Context seen by target':'Recorded output'}`,evidence=>{
        evidence.append(message(data,data.messages[reference.match.message],reference.match.location==='input'));
        const anchor=el('a','Open this turn in its phase');anchor.href=`#${prefix}turn-${reference.match.turn}`;evidence.append(anchor);
      }));
    }
  }),'full-original') : null;
  if(original) {original.id=localId('original-evaluation');container.before(original);}
  function onHash() {
    const match=location.hash.match(turnPattern);
    if (match) jump(Number(match[1]));
    if (focused && location.hash.startsWith('#phase-'+containerId.split('-').at(-1)+'-')) {
      const target=document.getElementById(location.hash.slice(1));
      if (target) {
        window._transectClearCardFilters?.(auditorContainerId);
        for (let node=target;node;node=node.parentElement) if (node.tagName==='DETAILS') node.open=true;
        requestAnimationFrame(()=>target.scrollIntoView({block:'start'}));
      }
    }
    if (original && location.hash===localHash('original-evaluation')) {original.open=true;original.scrollIntoView({block:'start'});}
    if (evidence.href?.startsWith('#') && location.hash===evidence.href) {
      const assessment=document.getElementById(evidence.href.slice(1));
      if (assessment) {assessment.open=true;assessment.scrollIntoView({block:'start'});}
    }
  }
  document.addEventListener('click',event=>{
    const href=event.target.closest('a')?.getAttribute('href');
    if (focused && href?.startsWith('#phase-'+containerId.split('-').at(-1)+'-') && href===location.hash) onHash();
    if (href===location.hash && ([evidence.href,localHash('original-evaluation')].includes(href) || turnPattern.test(href))) onHash();
  });
  window.addEventListener('hashchange',onHash);onHash();
  }
  for(const node of document.querySelectorAll('script[data-petri-config], #phase-transcript-config'))mount(node);
})();

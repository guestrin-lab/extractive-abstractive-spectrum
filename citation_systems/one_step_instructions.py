# Quoted OP generation
one_step_op_instruction_dict = {
    "Quoted": "Respond to the following query using word-for-word quotes from the sources provided below. Clearly indicate the quotes with quotation marks to avoid plagiarizing! Be concise in your response and focus on information that responds to the query. Do not refer to the sources in your response. In your response, cite individual sentences from the sources using the appropriate citation marker (e.g., [3]).",
    "Paraphrased":  "Respond to the following query by building off of the response below. Specifically, rephrase each sentence in the response using a more fluent and useful wording to convey the same information. In other words, paraphrase each sentence of the response as an improved new sentence, with respect to the query. Do not refer to the response in your revised response. In your response, cite individual sentences from the sources using the appropriate citation marker (e.g., [3]).",
    "Entailed": "Respond to the following query by building off of the response below. Specifically, rephrase and combine the sentences in the response by paraphrasing them, cutting out extraneous or redundant information, and simplifying details that are too fine-grained with respect to the question. Also, simplify premises to their logical conclusions to more directly answer the query. Do not refer to the response in your revised response. In your response, cite individual sentences from the sources using the appropriate citation marker (e.g., [3]).",
    "Abstractive": "Respond to the following query by building off of the response below. Specifically, rephrase and combine the sentences in the response by paraphrasing them, cutting out extraneous or redundant information, and simplifying details that are too fine-grained with respect to the question. Also, simplify premises to their logical conclusions to more directly answer the query. Most importantly, use accurate outside information to make the revised response a more useful answer to the query. If the provided response is not an accurate or useful answer to the query, then extensively revise it with accurate outside information. Respond in no more than about 100 words. Do not refer to the response in your revised response. In your response, cite individual sentences from the sources using the appropriate citation marker (e.g., [3])."
}

tg_one_step_system_prompt = """You are an attentive assistant. Follow all instructions carefully. For every response—regardless of task, input style, or complexity—rigorously maximize all of the following metrics and comply precisely with every operational requirement below. Treat atomic, unambiguous citation-claim mapping as a non-negotiable baseline for every claim, instruction, or revision. At every step, explicitly decompose, audit, and verify each atomic factual claim, detail, or data point for precise, one-to-one citation support, and transparently flag any inference or use of background knowledge.

**Citation Precision and Atomic Mapping:**  
- For every distinct factual claim, data point, or procedural step, place a citation marker *immediately after* (not at sentence ends or for claim clusters) the precise claim, and only if the cited source *directly* covers every granular component of that claim verbatim.
- Do not bundle or cluster citation markers for multi-claim sentences unless every cited source independently and exactly supports every part. If a sentence or segment contains multiple facts or attributes (names, dates, functions, relationships, transitions), split or restructure so each is mapped to its own citation.
- Never cite a source for a claim it implies, partially covers, or is only contextually related to; annotate unsupported elements explicitly as “No source provided” or “Based on general knowledge.”

**Coverage Auditing & Transparency:**  
- Systematically verify for each claim that it is fully supported by the source—never over-attribute a citation to more claims, qualifiers, or details than are present in the reference.
- For every output, break down and list each atomic claim with its assigned citation, verifying sentence-by-sentence sufficiency and auditability. Transparently annotate any coverage gaps, uncertainties, or inferential steps directly in the answer.
- When handling claims about transitions, overlapping states, or gradual changes, explicitly describe the transition period with source-aligned language; never compress or oversimplify multi-state timelines, and cite specifics for each period.

**Granular Handling of Logical Conclusions / Synthesis:**  
- Whenever a logical or comparative conclusion is derived (e.g., “Director X died before Y”), present each underlying atomic fact, each with its explicit citation, before making the comparison. Clearly signal comparison conclusions as a synthesis from the precedingly-cited facts, rather than citing the comparison unless it is stated verbatim in the source.  
- Mark any step, relationship, or causal claim not directly stated in the sources with “Based on general knowledge” or “No source provided,” and do not attach a citation for synthesis or inference steps unless the source presents them word-for-word.

**Context Clarity, Entity Referencing, and Utility:**  
- Always use explicit, unambiguous names for people, places, events, and relationships. Avoid layered possessives, ambiguous referents, or convoluted constructions.
- In comparison, timeline, and multi-claim answers, use parallel, skimmable construction: restate each entity and attribute in a clear, matched structure, never assuming context or relying on pronouns.
- Whenever quoting, paraphrasing, or explaining technical, domain-specific, or potentially unfamiliar terms for a general or young audience, immediately use the term as found in source and explain or define as necessary (unless forbidden by instructions).

**Fluency and Redundancy Elimination:**  
- Restructure as needed to avoid redundancy and ambiguity, synthesizing or consolidating similar or overlapping facts where possible. Never repeat or paraphrase similar facts from different sources without unique, additive value—each inclusion must be justified by new detail or context.
- Organize content into bottom-up, logically grouped, and skimmable segments or lists; structure stepwise or multi-claim answers as clear, parallel units for maximal auditability.

**Quotation and Paraphrase Protocol:**  
- Use quotation marks *only* for direct, verbatim source excerpts; mark all modifications, ellipses, or abridgments and cite accordingly. For non-verbatim paraphrasing, ensure full factual and semantic fidelity to the cited source and highlight the nature and limitations of paraphrase when relevant.

**Handling Absence of Sources and Gaps:**  
- Where a source is absent for any required fact, clearly flag the coverage gap as “No source provided” or “Based on general knowledge” immediately following the unsupported claim or segment. Never fabricate attributions or inferences; strictly do not add unsupported details.

**Bibliography, Source Transparency, and Formatting:**  
- Each in-text citation marker must unambiguously map to a checkable, explicit, and complete source reference (author, title, publication, date, URL if available) in a present, clearly formatted bibliography or source list.
- If repeated reference to a single source occurs, clarify the scope of every instance; if multiple atomic facts derive from different places in a single long document, indicate location (section, heading, paragraph) where feasible.

**Meta-Review / Audit Checklist (Non-Negotiable):**  
- Break every sentence into atomic factual claims and pair each with its own citation.
- Split or restructure complex/compound sentences as needed for exact, granular mapping; forbid citation clustering except for fully parallel support.
- Remove all redundant or near-duplicate statements; synthesize only if justified.
- Explicitly mark and explain any use of inference, partial support, or knowledge gaps within the narrative (not as an endnote).
- Audit for ambiguous, general, or over-extended citation; flag and fix all mismatches or gaps.
- Confirm every entity, time, place, and transition is explicit, auditable, and directly cited.
- Ensure every cited marker is matched correctly in a clean, itemized bibliography/source list.
- Check for and avoid pronouns or deictic referents unless context is bulletproof.
- Never compress, generalize, or elide technical or chronological nuance; if the transition is complex, state each period/fact exactly as sourced.

**Best Practice Examples:**  
- Always output atomic, auditably-mapped structure (e.g., “Alan Crosland, director of 'Greater Than Fame,' died in 1936 [X]. Gordon Douglas, director of 'The Pigskin Palooka,' died in 1993 [Y]. Therefore, Alan Crosland died earlier than Gordon Douglas.”).
- For timeline/transition scenarios, state: “In 1982, both copper and zinc-based pennies were produced [1][2]; after 1982, U.S. pennies were primarily made of zinc [1].”

By executing every requirement above with maximal rigor for every query, response, or revision, you will deliver output that supports perfect factual verification, transparency, and trusted audit for every fact, reasoned inference, and synthesis—no matter the complexity or instructions given."""

# Few-shot examples

def get_quoted_few_shot_examples(data_str):
    if data_str == 'nq':
        return f"""Instructions: {one_step_op_instruction_dict['Quoted']}    
Query: What's the life cycle of a frog?
Sub-questions: What's the life cycle of a frog?

Sources:

" [1] .S.
 [2] , Applied Ecology, Indiana University BloomingtonB.S.
 [3] , Biology and Chemistry, University of Illinois at Urbana-ChampaignLaura Klappenbach, M.S.
 [4] , is a science writer specializing in ecology, biology, and wildlife.
 [5] Learn about ourEditorial ProcessUpdated on August 11, 2019The life cycle of a frog consists of three stages: egg, larva, and adult.
 [6]  As the frog grows, it moves through these stages in a process known as metamorphosis."

" [7] But in a few species, parents remain with the eggs to look after them as they develop.
 [8]  As the fertilized eggs mature, the yolk in each egg splits into more and more cells and begins to take the form of a tadpole, the larva of a frog.
 [9]  Within one to three weeks, the egg is ready to hatch, and a tiny tadpole breaks free.
 [10] 03of 04Stage 2: Tadpole (Larva)Johner Images / Getty ImagesTadpoles, frogs' larvae, have rudimentary gills, a mouth, and a long tail.
 [11]  For the first week or two after the tadpole hatches, it moves very little.
 [12]  During this time, the tadpole absorbs the remaining yolk left over from the egg, which provides much-needed nourishment.
 [13]  After absorbing the yolk, the tadpole is strong enough to swim on its own.
 [14] Most tadpoles feed on algae and other vegetation, so they are considered herbivores.
 [15]  They filter material from the water as they swim or tear away bits of plant material.
 [16]  As the tadpole continues to grow, it begins to develop hind limbs."

" [17] The legless, water-bound tadpoles slowly metamorphose into frogs over the next 14 weeks.
 [18]  First, they grow back legs, then front legs too! Soon after, their body starts to change shape, and they’re able to start eating insects.
 [19]  Next, the tadpoles’ tails shrink away, and skin grows over their gills, as they develop lungs and eardrums! These are super important steps, as they prepare the tadpole for life on land.
 [20]  Once their gills and tails are gone forever, tadpoles undergo one last 24-hour push, where the metamorphosis completes.
 [21]  Once this stage is finished, the baby frogs emerge from the water as tiny adults!"

Response: "The life cycle of a frog consists of three stages: egg, larva, and adult" [5]. "As the frog grows, it moves through these stages in a process known as metamorphosis" [6], During the egg stage, "the yolk in each egg splits into more and more cells and begins to take the form of a tadpole, the larva of a frog" [8]. "Within one to three weeks, the egg is ready to hatch, and a tiny tadpole breaks free" [9]. "Tadpoles, frogs' larvae, have rudimentary gills, a mouth, and a long tail" [10]. As they grow, "tadpoles feed on algae and other vegetation" [14]. Eventually, "they grow back legs, then front legs too! Soon after, their body starts to change shape, and they’re able to start eating insects" [18]. "Next, the tadpoles’ tails shrink away, and skin grows over their gills, as they develop lungs and eardrums" [19].

Instructions: {one_step_op_instruction_dict['Quoted']}
Query: What are the benefits of hyaluronic acid for skin care?
Sub-questions: What are the benefits of hyaluronic acid for skin care?

Sources:

" [1] Hyaluronic acid is a remarkable substance because of all the benefits and uses it has in your body.
 [2]  Here are just a few of the benefits of hyaluronic acid: It helps things move smoothly.
 [3]  Hyaluronic acid helps your joints work like a well-oiled machine.
 [4]  It prevents pain and injury from bones grinding against each other.
 [5]  It helps keep things hydrated."

" [6] Hyaluronic acid is very good at retaining water.
 [7]  A quarter-teaspoon of hyaluronic acid holds about one and a half gallons of water.
 [8]  That’s why hyaluronic acid is often used for treating dry eyes.
 [9]  It’s also used in moisturizing creams, lotions, ointments and serums.
 [10]  It makes your skin flexible.
 [11]  Hyaluronic acid helps skin stretch and flex and reduces skin wrinkles and lines.
 [12]  Hyaluronic acid is also proven to help wounds heal faster and can reduce scarring."

Response: Hyaluronic acid "helps keep things hydrated" [5]. It "helps skin stretch and flex and reduces skin wrinkles and lines" [11]. "Hyaluronic acid is also proven to help wounds heal faster and can reduce scarring" [12].

Instructions: {one_step_op_instruction_dict['Quoted']}
Query: What is a heirloom tomato?
Sub-questions: What is a heirloom tomato?

Sources:

" [1] They are classified as family heirlooms, commercial heirlooms, mystery heirlooms, or created heirlooms.
 [2]  They usually have a shorter shelf life and are less disease resistant than hybrids.
 [3]  They are grown for various reasons: for food, historical interest, access to wider varieties, and by people who wish to save seeds from year to year, as well as for their taste.
 [4] [1]Taste[edit]Many heirloom tomatoes are sweeter and lack a genetic mutation that gives tomatoes a uniform red color at the cost of the fruit's taste.
 [5] [2] Varieties bearing that mutation which have been favored by industry since the 1940s – that is, tomatoes which are not heirlooms – feature fruits with lower levels of carotenoids and a decreased ability to make sugar within the fruit."

" [6] True heirlooms must be 'old' because they have been a variety for anywhere between 25 and 50 years.
 [7]  The seeds, like furniture or jewelry, must also be passed down from generation to generation.
 [8]  There is no exact blueprint for the color and shape of an heirloom tomato, so long as it meets the three characteristics.
 [9]  There are over 100 different types of heirloom tomatoes in the united states and an estimated 3,000 globally, which are suitable for very different climates and growing regions."

Response: "Many heirloom tomatoes are sweeter and lack a genetic mutation that gives tomatoes a uniform red color at the cost of the fruit's taste" [4]. "True heirlooms must be 'old' because they have been a variety for anywhere between 25 and 50 years" [6].

Instructions: {one_step_op_instruction_dict['Quoted']}
Query: What's the flavor of juicy fruit gum?
Sub-questions: What's the flavor of juicy fruit gum?

Sources:

" [1] A Juicy Fruit wrapper from 1946 reportedly advertised that the gum had 'a fascinating artificial flavor,' which doesn't really provide much clarity.
 [2]  In 2002, one curious fan took their question to the company itself and received the following statement in response, 'I'm afraid we can't be very specific because, for competitive reasons, we consider our Juicy Fruit flavoring formula to be a trade secret.
 [3]  I can, however, tell you that the mixture of fruit flavors in Juicy Fruit is comprised of predominately lemon, orange, pineapple, and banana notes.'"

Response: "the mixture of fruit flavors in Juicy Fruit is comprised of predominately lemon, orange, pineapple, and banana notes" [3].
"""
    elif data_str == 'eli5_nq':
        return f"""Instructions: {one_step_op_instruction_dict['Quoted']}

Instructions: {one_step_op_instruction_dict['Quoted']}
Query: Explain to a third-grader: what's the life cycle of a frog?
Sub-questions: Explain to a third-grader: what's the life cycle of a frog?

Sources:

" [1] .S.
 [2] , Applied Ecology, Indiana University BloomingtonB.S.
 [3] , Biology and Chemistry, University of Illinois at Urbana-ChampaignLaura Klappenbach, M.S.
 [4] , is a science writer specializing in ecology, biology, and wildlife.
 [5] Learn about ourEditorial ProcessUpdated on August 11, 2019The life cycle of a frog consists of three stages: egg, larva, and adult.
 [6]  As the frog grows, it moves through these stages in a process known as metamorphosis."

" [7] But in a few species, parents remain with the eggs to look after them as they develop.
 [8]  As the fertilized eggs mature, the yolk in each egg splits into more and more cells and begins to take the form of a tadpole, the larva of a frog.
 [9]  Within one to three weeks, the egg is ready to hatch, and a tiny tadpole breaks free.
 [10] 03of 04Stage 2: Tadpole (Larva)Johner Images / Getty ImagesTadpoles, frogs' larvae, have rudimentary gills, a mouth, and a long tail.
 [11]  For the first week or two after the tadpole hatches, it moves very little.
 [12]  During this time, the tadpole absorbs the remaining yolk left over from the egg, which provides much-needed nourishment.
 [13]  After absorbing the yolk, the tadpole is strong enough to swim on its own.
 [14] Most tadpoles feed on algae and other vegetation, so they are considered herbivores.
 [15]  They filter material from the water as they swim or tear away bits of plant material.
 [16]  As the tadpole continues to grow, it begins to develop hind limbs."

" [17] The legless, water-bound tadpoles slowly metamorphose into frogs over the next 14 weeks.
 [18]  First, they grow back legs, then front legs too! Soon after, their body starts to change shape, and they’re able to start eating insects.
 [19]  Next, the tadpoles’ tails shrink away, and skin grows over their gills, as they develop lungs and eardrums! These are super important steps, as they prepare the tadpole for life on land.
 [20]  Once their gills and tails are gone forever, tadpoles undergo one last 24-hour push, where the metamorphosis completes.
 [21]  Once this stage is finished, the baby frogs emerge from the water as tiny adults!"

Response: "The life cycle of a frog consists of three stages: egg, larva, and adult" [5]. "As the frog grows, it moves through these stages in a process known as metamorphosis" [6], During the egg stage, "the yolk in each egg splits into more and more cells and begins to take the form of a tadpole, the larva of a frog" [8]. "Within one to three weeks, the egg is ready to hatch, and a tiny tadpole breaks free" [9]. "Tadpoles, frogs' larvae, have rudimentary gills, a mouth, and a long tail" [10]. As they grow, "tadpoles feed on algae and other vegetation" [14]. Eventually, "they grow back legs, then front legs too! Soon after, their body starts to change shape, and they’re able to start eating insects" [18]. "Next, the tadpoles’ tails shrink away, and skin grows over their gills, as they develop lungs and eardrums" [19].

Instructions: {one_step_op_instruction_dict['Quoted']}
Query: Explain to a third-grader: what are the benefits of hyaluronic acid for skin care?
Sub-questions: Explain to a third-grader: what are the benefits of hyaluronic acid for skin care?

Sources:

" [1] Hyaluronic acid is a remarkable substance because of all the benefits and uses it has in your body.
 [2]  Here are just a few of the benefits of hyaluronic acid: It helps things move smoothly.
 [3]  Hyaluronic acid helps your joints work like a well-oiled machine.
 [4]  It prevents pain and injury from bones grinding against each other.
 [5]  It helps keep things hydrated."

" [6] Hyaluronic acid is very good at retaining water.
 [7]  A quarter-teaspoon of hyaluronic acid holds about one and a half gallons of water.
 [8]  That’s why hyaluronic acid is often used for treating dry eyes.
 [9]  It’s also used in moisturizing creams, lotions, ointments and serums.
 [10]  It makes your skin flexible.
 [11]  Hyaluronic acid helps skin stretch and flex and reduces skin wrinkles and lines.
 [12]  Hyaluronic acid is also proven to help wounds heal faster and can reduce scarring."

Response: Hyaluronic acid "helps keep things hydrated" [5]. It "helps skin stretch and flex and reduces skin wrinkles and lines" [11]. "Hyaluronic acid is also proven to help wounds heal faster and can reduce scarring" [12].

Instructions: {one_step_op_instruction_dict['Quoted']}
Query: Explain to a third-grader: what is a heirloom tomato?
Sub-questions: Explain to a third-grader: what is a heirloom tomato?

Sources:

" [1] They are classified as family heirlooms, commercial heirlooms, mystery heirlooms, or created heirlooms.
 [2]  They usually have a shorter shelf life and are less disease resistant than hybrids.
 [3]  They are grown for various reasons: for food, historical interest, access to wider varieties, and by people who wish to save seeds from year to year, as well as for their taste.
 [4] [1]Taste[edit]Many heirloom tomatoes are sweeter and lack a genetic mutation that gives tomatoes a uniform red color at the cost of the fruit's taste.
 [5] [2] Varieties bearing that mutation which have been favored by industry since the 1940s – that is, tomatoes which are not heirlooms – feature fruits with lower levels of carotenoids and a decreased ability to make sugar within the fruit."

" [6] True heirlooms must be 'old' because they have been a variety for anywhere between 25 and 50 years.
 [7]  The seeds, like furniture or jewelry, must also be passed down from generation to generation.
 [8]  There is no exact blueprint for the color and shape of an heirloom tomato, so long as it meets the three characteristics.
 [9]  There are over 100 different types of heirloom tomatoes in the united states and an estimated 3,000 globally, which are suitable for very different climates and growing regions."

Response: "Many heirloom tomatoes are sweeter and lack a genetic mutation that gives tomatoes a uniform red color at the cost of the fruit's taste" [4]. "True heirlooms must be 'old' because they have been a variety for anywhere between 25 and 50 years" [6].

Instructions: {one_step_op_instruction_dict['Quoted']}
Query: Explain to a third-grader: what's the flavor of juicy fruit gum?
Sub-questions: Explain to a third-grader: what's the flavor of juicy fruit gum?

Sources:

" [1] A Juicy Fruit wrapper from 1946 reportedly advertised that the gum had 'a fascinating artificial flavor,' which doesn't really provide much clarity.
 [2]  In 2002, one curious fan took their question to the company itself and received the following statement in response, 'I'm afraid we can't be very specific because, for competitive reasons, we consider our Juicy Fruit flavoring formula to be a trade secret.
 [3]  I can, however, tell you that the mixture of fruit flavors in Juicy Fruit is comprised of predominately lemon, orange, pineapple, and banana notes.'"

Response: "the mixture of fruit flavors in Juicy Fruit is comprised of predominately lemon, orange, pineapple, and banana notes."
"""
    elif data_str == 'mash':
        return f"""Instructions: {one_step_op_instruction_dict['Quoted']}

Instructions: {one_step_op_instruction_dict['Quoted']}
Query: What can you do to prevent migraine with brainstem aura?
Sub-questions: What can you do to prevent migraine with brainstem aura?

Sources:

" [1] Symptoms of MBA can also seem like the signs of other more serious conditions, like seizure disorders, stroke, meningitis, or brain tumors.
 [2]  To rule those out, you'll need to see a brain doctor, called a neurologist.
 [3]  He'll give you a thorough exam and ask you questions about your symptoms.
 [4]  He'll may also use tests like MRI, CT scans, and nerve tests to see what's causing your symptoms.
 [5]  Treatments for MBA generally aim to relieve symptoms of pain and nausea.
 [6]  You might take pain relievers such as acetaminophen, ibuprofen, and naproxen, as well as nausea medicines such as chlorpromazine, metoclopramide, and prochlorperazine.
 [7]  Your doctor may prescribe a medication that treats regular migraines, such as triptans.
 [8]  To keep from getting a MBA,  it helps to avoid the things that usually cause one."

" [9] Keep a journal of your attacks so you can figure out the things that trigger them.
 [10]  It also helps to live a healthy lifestyle.
 [11]  That means you need to:  Get enough sleep.
 [12]   Limit your stress.
 [13]   Exercise daily.
 [14]  Diet can also affect migraines.
 [15]  Do these things:  Eat a balanced diet.
 [16]   Avoid drugs and alcohol.
 [17]   Don't skip meals.
 [18]   Limit caffeine.
 [19]   Avoid any foods that have been triggers.
 [20]  Some common food triggers include: Dairy Wheat Chocolate Eggs Rye Tomatoes Oranges."

" [21] Along with these changes,  If you don't respond to other treatments and you have 4 or more migraine days a month, your doctor may suggest preventive medicines.
 [22]  You can take these regularly to reduce the severity or frequency of the headaches.
 [23]  These include seizure medicines, blood pressure medicines (like beta blockers and calcium channel blockers), and some antidepressants.
 [24]  CGRP inhibitors are a new class of preventive medicine that your doctor may recommend if other medicines don't help."

Response: To prevent migraine with brainstem aura, "it helps to avoid the things that usually cause one" [8]. This includes being sure to "Get enough sleep" [11], "Limit your stress" [12], and "Exercise daily" [13]. Additionally, "Eat a balanced diet" [15], "Avoid drugs and alcohol" [16], "Don't skip meals" [17], and "Limit caffeine" [18]. "Avoid any foods that have been triggers" [19]. Furthermore, "If you don't respond to other treatments and you have 4 or more migraine days a month, your doctor may suggest preventive medicines" [21], including "seizure medicines, blood pressure medicines (like beta blockers and calcium channel blockers), and some antidepressants" [23]. Also, "CGRP inhibitors are a new class of preventive medicine" [24].

Instructions: {one_step_op_instruction_dict['Quoted']}
Query: What should I do if my child has a biting injury?
Sub-questions: What should I do if my child has a biting injury?

Sources:

" [1] If your child is prone to biting, keep an eye on any playmates and step in when an altercation appears to be brewing.
 [2]  Even with your best prevention efforts, biting incidents might still occur.
 [3]  When your child bites, firmly let your child know that this behavior is not acceptable by saying, 'No.
 [4]  We don't bite!'
 [5]  Explain that biting hurts the other person.
 [6]  Then remove your child from the situation and give the child time to calm down.
 [7]  You might have heard from other parents that if your child bites you, bite your child back.
 [8]  This isn't good advice.
 [9]  Children learn by imitation.
 [10]  If you bite your child, the child is going to get the impression that this behavior is acceptable and he or she will be more likely to do it again.
 [11]  The same goes for hitting a child for biting.
 [12]  If you are unable to get your child to stop biting, the behavior could begin to have an impact on school and relationships.
 [13]  You or another adult might have to closely supervise interactions between your child and other kids.
 [14]  When biting becomes a habit or continues past age 4 or 5, it might stem from a more serious emotional problem.
 [15]  Talk to your child's health care provider, or enlist the help of a child psychologist or therapist."

" [16] [1] The first thing to do for any biting injury is to wash the area with soap and water.
 [17]  [2] Even little teeth can break the skin.
 [18]  [3] If the bite is bleeding and the wound appears to be deep, call your child's doctor.
 [19]  [4] The bite may need medical treatment, which could include antibiotics or a tetanus shot, or both."

Response: "The first thing to do for any biting injury is to wash the area with soap and water" [16]. "Even little teeth can break the skin" [17]. "If the bite is bleeding and the wound appears to be deep, call your child's doctor" [18]. "The bite may need medical treatment, which could include antibiotics or a tetanus shot, or both" [19].

Instructions: {one_step_op_instruction_dict['Quoted']}
Query: Can I give cough and cold medicines to children under 4?
Sub-questions: Can I give cough and cold medicines to children under 4?

Sources:

" [1] There are over 200 cold viruses ready to lay your family low with nasal congestion, sore throat, and cough.
 [2]  Combat cold symptoms with these home remedies.
 [3]  Get rest.
 [4]  Rest helps the body focus its energy on getting well, so keep kids home from school to keep them warm, and well-rested."

" [5] Drink lots of fluids.
 [6]  Replenish liquids lost from fever, vomiting and diarrhea.
 [7]  Fluids also help loosen mucus.
 [8]  Use a humidifier.
 [9]  A humidifier in your child's room can keep the air moist and break up nasal and chest congestion.
 [10]  Talk to your pediatrician before giving OTC cold and cough medicines.
 [11]  These medicines should not be given to children under 4 years of age, according to the FDA and the drug makers.
 [12]  Also, evidence indicates medicines such as antihistamines, decongestants, and cough medicines don't really help, but they could pose a small risk of serious side effects."

Response: "Talk to your pediatrician before giving OTC cold and cough medicines" [10]. "These medicines should not be given to children under 4 years of age, according to the FDA and the drug makers" [11].

Instructions: {one_step_op_instruction_dict['Quoted']}
Query: How do I treat a blister?
Sub-questions: How do I treat a blister?

Sources:

" [1] Most blisters caused by friction or minor burns do not require a doctor's care.
 [2]   New skin will form underneath the affected area and the fluid is simply absorbed."

" [3] Do not puncture a blister unless it is large, painful, or likely to be further irritated.
 [4]   The fluid-filled blister keeps the underlying skin clean, which prevents infection and promotes healing.
 [5]   But if you need to pop a blister or it pops by itself: Use a sterilized needle (to sterilize it, put the point or edge in a flame until it is red hot, or rinse it in alcohol).
 [6]   Wash your hands and the area thoroughly, then make a small hole; the fluid will drain on its own.
 [7]   If the fluid is white or yellow, the blister may be infected and needs medical attention.
 [8]   Do not remove the skin over a broken blister.
 [9]   The new skin underneath needs this protective cover.
 [10]   Apply an antibiotic ointment or cream.
 [11]   Look for signs of infection to develop, including pus drainage, red or warm skin surrounding the blister, or red streaks leading away from the blister."

" [12] Wear work gloves.
 [13]  Jobs you do only occasionally, such as shoveling snow or raking leaves, are great for raising a blister or two.
 [14]  Wear shoes that fit and are comfortable.
 [15]  New shoes should be broken in gradually.
 [16]  Wear padded socks or put some adhesive padding where your foot is rubbing."

Response: "Most blisters caused by friction or minor burns do not require a doctor's care" [1]. "New skin will form underneath the affected area and the fluid is simply absorbed" [2]. "Do not puncture a blister unless it is large, painful, or likely to be further irritated" [3]. "The fluid-filled blister keeps the underlying skin clean, which prevents infection and promotes healing" [4]. "But if you need to pop a blister or it pops by itself: Use a sterilized needle (to sterilize it, put the point or edge in a flame until it is red hot, or rinse it in alcohol)." [5]. "Wash your hands and the area thoroughly, then make a small hole; the fluid will drain on its own" [6]. "If the fluid is white or yellow, the blister may be infected and needs medical attention" [7]. "Do not remove the skin over a broken blister" [8]. "The new skin underneath needs this protective cover" [9]. "Apply an antibiotic ointment or cream" [10]. "Look for signs of infection to develop, including pus drainage, red or warm skin surrounding the blister, or red streaks leading away from the blister" [11].
"""
    elif data_str == 'multihop':
        return f"""Instructions: {one_step_op_instruction_dict['Quoted']}
Query: Are the director of film Move (1970 Film) and director of film Méditerranée (1963 Film) from the same country?
Sub-questions: Who is the director of the film Move (1970 film)? What country are they from? Who is the director of the film Méditerranée (1963 Film)? What country are they from?
Sources:

" [1] Stuart Rosenberg (August 11, 1927 – March 15, 2007) was an American film and television director whose motion pictures include 'Cool Hand Luke' (1967), 'Voyage of the Damned' (1976), 'The Amityville Horror' (1979), and 'The Pope of Greenwich Village' (1984). [2] He was noted for his work with actor Paul Newman."

" [3] 'Méditerranée' is a 1963 French experimental film directed by Jean-Daniel Pollet with assistance from Volker Schlöndorff. [4] It was written by Philippe Sollers and produced by Barbet Schroeder, with music by Antione Duhamel. [5] The 45 minute film is cited as one of Pollet's most influential films, which according to Jonathan Rosenbaum directly influenced Jean-Luc Goddard's 'Contempt', released later the same year. [6] Footage for the film was shot around the Mediterranean, including at a Greek temple, a Sicilian garden, the sea, and also features a fisherman, a bullfighter, and a girl on an operating table."

" [7] Move is a 1970 American comedy film starring Elliott Gould, Paula Prentiss and Geneviève Waïte, and directed by Stuart Rosenberg. [8] The screenplay was written by Joel Lieber and Stanley Hart, adapted from a novel by Lieber."

" [9] Jean-Daniel Pollet (1936–2004) was a French film director and screenwriter who was most active in the 1960s and 1970s. [10] He was associated with two approaches to filmmaking: comedies which blended burlesque and melancholic elements, and poetic films based on texts by writers such as the French poet Francis Ponge."

Response: "Move is a 1970 American comedy film starring Elliott Gould, Paula Prentiss and Geneviève Waïte, and directed by Stuart Rosenberg" [7]. "Stuart Rosenberg (August 11, 1927 – March 15, 2007) was an American film and television director" [1]. However, "'Méditerranée' is a 1963 French experimental film directed by Jean-Daniel Pollet" [3] and "Jean-Daniel Pollet (1936–2004) was a French film director and screenwriter" [9].

Instructions: {one_step_op_instruction_dict['Quoted']}
Query: What nationality is the director of film Wedding Night In Paradise (1950 Film)?
Sub-questions:Who is the director of Wedding Night In Paradise (1950 Film)? What is their nationality?
Sources:

" [1] Wedding Night in Paradise  is a 1950 West German musical comedy film directed by Géza von Bolváry and starring Johannes Heesters, Claude Farell and Gretl Schörg. [2] It is an operetta film, based on the 1942 stage work of the same title. [3] The film's sets were designed by Paul Markwitz and Fritz Maurischat."

" [4] Géza von Bolváry (full name Géza Maria von Bolváry-Zahn; 26 December 1897 – 10 August 1961) was a Hungarian actor, screenwriter and film director, who worked principally in Germany and Austria."

Response: "Wedding Night in Paradise  is a 1950 West German musical comedy film directed by Géza von Bolváry" [1]. "Géza von Bolváry (full name Géza Maria von Bolváry-Zahn; 26 December 1897 – 10 August 1961) was a Hungarian actor, screenwriter and film director" [4].

Instructions: {one_step_op_instruction_dict['Quoted']}
Query: When did the performer of song A Pretty Girl Milking Her Cow die?
Sub-questions: Who performed the song A Pretty Girl Milking Her Cow? When did they die?
Sources:

" [1] Judy Garland (born Frances Ethel Gumm; June 10, 1922 – June 22, 1969) was an American actress, singer and dancer. [2] During a career that spanned 45 years, she attained international stardom as an actress in both musical and dramatic roles, as a recording artist, and on the concert stage."

" [3] A Pretty Girl Milking Her Cow (Cailin Deas Crúite na mBó in the Irish language) is a traditional 18th-century Irish ballad. [4] The English version is attributed to Thomas Moore (1779–1852). [5] Originally sung in Irish Gaelic, the song was popular through the early 20th century. [6] It enjoyed a revival when an updated swing version sung by Irish-American singer/actress Judy Garland was featured in the 1940 film Little Nellie Kelly. [7] The updated version is true to the original musical air, and incorporated original lyrics by MGM Musical Director Roger Edens, and featured Garland singing the song to George Murphy using some of the original Gaelic lyrics in the first chorus."

Response: "Judy Garland (born Frances Ethel Gumm; June 10, 1922 – June 22, 1969) was an American actress, singer and dancer" [1]. A Pretty Girl Milking Her Cow "enjoyed a revival when an updated swing version sung by Irish-American singer/actress Judy Garland was featured in the 1940 film Little Nellie Kelly" [3].

Instructions: {one_step_op_instruction_dict['Quoted']}
Query: Do both films The Falcon (Film) and Valentin The Good have the directors from the same country?
Sub-questions: Who is the director of the film Falcon? What country are they from? Who is the director of Valentin The Good? What country are they from?
Sources:

" [1] Vatroslav Mimica( born 25 June 1923) is a Croatian film director and screenwriter. [2] Born in Omiš, Mimica had enrolled at the University of Zagreb School of Medicine before the outbreak of World War II. [3] In 1942 he joined Young Communist League of Yugoslavia( SKOJ) and in 1943 he went on to join the Yugoslav Partisans, becoming a member of their medical units. [4] After the war Mimica wrote literary and film reviews, and his career in filmmaking began in 1950 when he became the director of the Jadran Film production studio."

" [5] Martin Frič( 29 March 1902 – 26 August 1968) was a Czech film director, screenwriter and actor. [6] He had more than 100 directing credits between 1929 and 1968, including feature films, shorts and documentary films. [7] Throughout his life, Frič struggled with alcoholism. [8] On the day of the Warsaw Pact invasion of Czechoslovakia in 1968, he attempted suicide, after battling cancer. [9] He died in the hospital five days later."

" [10] Valentin the Good is a 1942 Czech comedy film directed by Martin Frič."

" [11] Banović Strahinja( Serbian Cyrillic:' Бановић Страхиња', released internationally as The Falcon) is a 1981 Yugoslavian- German adventure film written and directed by Vatroslav Mimica based on Strahinja Banović, a hero of Serbian epic poetry. [12] It entered the section' Officina Veneziana' at the 38th Venice International Film Festival."

Response: "Banović Strahinja( Serbian Cyrillic:' Бановић Страхиња', released internationally as The Falcon) is a 1981 Yugoslavian- German adventure film written and directed by Vatroslav Mimica" [11]. "Vatroslav Mimica( born 25 June 1923) is a Croatian film director and screenwriter" [1]. "Valentin the Good is a 1942 Czech comedy film directed by Martin Frič" [10] and "Martin Frič( 29 March 1902 – 26 August 1968) was a Czech film director, screenwriter and actor" [5].

Instructions: {one_step_op_instruction_dict['Quoted']}
Query: What is the date of birth of Mina Gerhardsen's father?
Sub-questions: Who is Mina Gerhardsen's father? What date was he born?
Sources:

" [1] Mina Gerhardsen (born 14 September 1975) is a Norwegian politician for the Labour Party. [2] She is the daughter of Rune Gerhardsen and Tove Strand, and granddaughter of Einar Gerhardsen. [3] She is married to Eirik Øwre Thorshaug. [4] She led the Oslo branch of Natur og Ungdom from 1993 to 1995, and was deputy leader of the Workers' Youth League in Oslo in 1997. [5] She took the cand.mag. degree at the University of Oslo in 1998, and also has master's degrees in pedagogy from 2000 and human geography from 2003. [6] From 1999 to 2002 she worked part-time as a journalist in 'Dagsavisen' and 'Dagbladet'. [7] She then worked in the Norwegian Red Cross from 2002 to 2004, except for a period from 2003 to 2004 as a journalist in 'Mandag Morgen'. [8] She was hired as a political advisor in the Norwegian Office of the Prime Minister in 2005, when Stoltenberg's Second Cabinet assumed office. [9] In 2009 she was promoted to State Secretary. [10] In 2011 she changed to the Ministry of Culture."

" [11] Rune Gerhardsen (born 13 June 1946) is a Norwegian politician, representing the Norwegian Labour Party. [12] He is a son of Werna and Einar Gerhardsen, and attended Oslo Cathedral School. [13] He chaired the Workers' Youth League from 1973 to 1975 and chaired the City Government of Oslo from 1992 to 1997. [14] He chaired the Norwegian Skating Association from 1986 to 1990 and 2001 to 2003 and also 2013 to 2017."

Response: "Mina Gerhardsen (born 14 September 1975) is a Norwegian politician for the Labour Party" [1]. "She is the daughter of Rune Gerhardsen and Tove Strand" [2]. "Rune Gerhardsen (born 13 June 1946) is a Norwegian politician" [11].

Instructions: {one_step_op_instruction_dict['Quoted']}
Query: When is the composer of film Sruthilayalu 's birthday?
Sub-questions: Who was the composer for film Sruthilayalu? When is their birthday?
Sources:

" [1] K. V. Mahadevan Krishnankoil Venkadachalam Mahadevan (14 March 1918 – 21 June 2001) was an Indian composer, singer-songwriter, music producer, and musician known for his works in Tamil cinema, Telugu cinema, Kannada cinema, and Malayalam cinema. [2] He is best known for his contributions in works such as 'Manchi Manasulu' (1962), 'Lava Kusa' (1963), 'Thiruvilaiyadal' (1965), 'Saraswathi Sabatham' (1966), 'Kandan Karunai' (1967), 'Thillana Mohanambal' (1968), 'Adimai Penn' (1969), 'Balaraju Katha' (1970), 'Athiparasakthi' (1971), 'Sankarabharanam' (1979), 'Saptapadi' (1981), 'Sirivennela' (1986), 'Sruthilayalu' (1987), 'Pelli Pustakam' (1991), and 'Swathi Kiranam' (1992)."

" [3] Sruthilayalu is a 1987 Indian Telugu-language musical drama film, written and directed by K. Viswanath. [4] The film stars Rajasekhar and Sumalata with soundtrack composed by K. V. Mahadevan. [5] The film garnered Nandi Awards for Best feature film; Best direction, and a Filmfare Award for Best Director – Telugu. [6] The film was premiered at the International Film Festival of India, and AISFM Film Festival. [7] The film was dubbed in Tamil as 'Isaikku Oru Koil'."

Response: "Sruthilayalu is a 1987 Indian Telugu-language musical drama film, written and directed by K. Viswanath. The film stars Rajasekhar and Sumalata with soundtrack composed by K. V. Mahadevan" [3]. "Krishnankoil Venkadachalam Mahadevan (14 March 1918 – 21 June 2001) was an Indian composer, singer-songwriter, music producer, and musician known for his works in Tamil cinema, Telugu cinema, Kannada cinema, and Malayalam cinema" [1].

Instructions: {one_step_op_instruction_dict['Quoted']}
Query: When did Fatima Bint Mubarak Al Ketbi's husband die?
Sub-questions: Who is Fatima Bint Mubarak Al Ketbi's husband? When did they die?
Sources:

" [1] Sheikh Zayed bin Sultan Al Nahyan ; 6 May 1918 – 2 November 2004) was the ruler of Abu Dhabi for more than 30 years (6 August 1966 – 2 November 2004). [2] He was the founding father and the principal driving force behind the formation of the United Arab Emirates."

" [3] Fatima bint Mubarak Al Ketbi is the third wife of Sheikh Zayed bin Sultan Al Nahyan, the founder and inaugural president of United Arab Emirates, and late emir (ruler) of Abu Dhabi. [4] She is referred to as the mother of sheikhs and as the Mother of the UAE."

Response: "Fatima bint Mubarak Al Ketbi is the third wife of Sheikh Zayed bin Sultan Al Nahyan" [3]. "Sheikh Zayed bin Sultan Al Nahyan ; 6 May 1918 – 2 November 2004) was the ruler of Abu Dhabi for more than 30 years (6 August 1966 – 2 November 2004)" [1].

Instructions: {one_step_op_instruction_dict['Quoted']}
Query: Who is the mother-in-law of Andrew Murray (Scottish Soldier)?
Sub-questions: Who is Andrew Murray's spouse? Who is their mother?
Sources:

" [1] Christina Bruce (c. 1278 – 1356/1357), also known as Christina de Brus, was a daughter of Marjorie, Countess of Carrick, and her husband, Robert de Brus, 'jure uxoris' Earl of Carrick, as well as a sister of Robert the Bruce, King of Scots. [2] It is presumed that she and her siblings were born at Turnberry Castle in Carrick."

" [3] Sir Andrew Murray (1298–1338), also known as Sir Andrew Moray, or Sir Andrew de Moray, was a Scottish military and political leader who supported David II of Scotland against Edward Balliol and King Edward III of England during the so-called Second War of Scottish Independence. [4] He held the lordships of Avoch and Petty in north Scotland, and Bothwell in west-central Scotland. [5] In 1326 he married Christina Bruce, a sister of King Robert I of Scotland."

Response: Sir Andrew Murray "married Christina Bruce, a sister of King Robert I of Scotland" [5]. "Christina Bruce (c. 1278 – 1356/1357), also known as Christina de Brus, was a daughter of Marjorie, Countess of Carrick, and her husband, Robert de Brus, 'jure uxoris' Earl of Carrick, as well as a sister of Robert the Bruce, King of Scots" [1].
"""
    else:
        print('Quoted few-shot examples not found for this dataset.')
        exit()

def get_paraphrased_few_shot_examples(data_str):
    if data_str == 'nq':
        return f"""Instructions: {one_step_op_instruction_dict['Paraphrased']}
Query: What's the life cycle of a frog?
Response: The life cycle of a frog consists of three stages: egg, larva, and adult [1]. As the frog grows, it moves through these stages in a process known as metamorphosis [2]. During the egg stage, the yolk in each egg splits into more and more cells and begins to take the form of a tadpole, the larva of a frog [3]. Within one to three weeks, the egg is ready to hatch, and a tiny tadpole breaks free [4]. In the larval stage, [t]adpoles, frogs' larvae, have rudimentary gills, a mouth, and a long tail [5]. As they grow, tadpoles feed on algae and other vegetation, and eventually, they grow back legs, then front legs too [6][7]! Soon after, their body starts to change shape, and they’re able to start eating insects [8]. Next, the tadpoles’ tails shrink away, and skin grows over their gills, as they develop lungs and eardrums [9]!
Paraphrased Response: The stages of a frog’s life cycle are egg, larva, and adult [1]. Growing through these stages is a process called metamorphosis [2]. In a frog egg, the yolk splits into many cells which are the beginnings of a tadpole, or a frog larva [3]. In one to three weeks, a small tadpole hatches from the egg [4]. During the larval stage, tadpoles, or frogs’ larvae, are equipped with simple gills, a mouth, and a lengthy tail [5]. As they develop, tadpoles consume plants and algae and then grow back legs, followed by front legs [6][7]. Shortly afterward, their body begins to shift in shape and they gain the ability to consume insects [8]. Then, the tails of the tadpoles grow smaller and skin develops over their gills while they grow eardrums and lungs [9].

Instructions: {one_step_op_instruction_dict['Paraphrased']}
Query: What are the benefits of hyaluronic acid for skin care?
Response: Hyaluronic acid helps keep things hydrated [1]. It helps skin stretch and flex and reduces skin wrinkles and lines [2]. Hyaluronic acid is also proven to help wounds heal faster and can reduce scarring [3][4].
Paraphrased Response: Hyaluronic acid helps to retain moisture [1]. It makes it easier for skin to be flexible and diminishes skin lines and wrinkles [2]. Hyaluronic acid has been shown to decrease the recovery time of wounds and can lessen scarring [3][4].

Instructions: {one_step_op_instruction_dict['Paraphrased']}
Query: What is a heirloom tomato?
Response: Many heirloom tomatoes are sweeter and lack a genetic mutation that gives tomatoes a uniform red color at the cost of the fruit's taste [1]. True heirlooms must be 'old' because they have been a variety for anywhere between 25 and 50 years [2].
Paraphrased Response: Unlike tomatoes that carry a genetic mutation that makes them look consistently red at the expense of their flavor, many heirloom tomatoes taste sweeter [1]. Genuine heirloom tomatoes have to be old due to being a variety over the last 25 to 50 years [2].

Instructions: {one_step_op_instruction_dict['Paraphrased']}
Query: What's the flavor of juicy fruit gum?
Response: the mixture of fruit flavors in Juicy Fruit is comprised of predominately lemon, orange, pineapple, and banana notes [1].
Paraphrased Response: Juicy Fruit flavor is a blend of mostly orange, lemon, banana, and pineapple flavors [1].
"""
    elif data_str == 'eli5_nq':
        return f"""Instructions: {one_step_op_instruction_dict['Paraphrased']}
Query: What's the life cycle of a frog?
Response: The life cycle of a frog consists of three stages: egg, larva, and adult [1]. As the frog grows, it moves through these stages in a process known as metamorphosis [2]. During the egg stage, the yolk in each egg splits into more and more cells and begins to take the form of a tadpole, the larva of a frog [3]. Within one to three weeks, the egg is ready to hatch, and a tiny tadpole breaks free [4]. In the larval stage, [t]adpoles, frogs' larvae, have rudimentary gills, a mouth, and a long tail [5]. As they grow, tadpoles feed on algae and other vegetation, and eventually, they grow back legs, then front legs too [6][7]! Soon after, their body starts to change shape, and they’re able to start eating insects [8]. Next, the tadpoles’ tails shrink away, and skin grows over their gills, as they develop lungs and eardrums [9]!
Paraphrased Response: The stages of a frog’s life cycle are egg, larva, and adult [1]. Growing through these stages is a process called metamorphosis [2]. In a frog egg, the yolk splits into many cells which are the beginnings of a tadpole, or a frog larva [3]. In one to three weeks, a small tadpole hatches from the egg [4]. During the larval stage, tadpoles, or frogs’ larvae, are equipped with simple gills, a mouth, and a lengthy tail [5]. As they develop, tadpoles consume plants and algae and then grow back legs, followed by front legs [6][7]. Shortly afterward, their body begins to shift in shape and they gain the ability to consume insects [8]. Then, the tails of the tadpoles grow smaller and skin develops over their gills while they grow eardrums and lungs [9].

Instructions: {one_step_op_instruction_dict['Paraphrased']}
Query: What are the benefits of hyaluronic acid for skin care?
Response: Hyaluronic acid helps keep things hydrated [1]. It helps skin stretch and flex and reduces skin wrinkles and lines [2]. Hyaluronic acid is also proven to help wounds heal faster and can reduce scarring [3][4].
Paraphrased Response: Hyaluronic acid helps to retain moisture [1]. It makes it easier for skin to be flexible and diminishes skin lines and wrinkles [2]. Hyaluronic acid has been shown to decrease the recovery time of wounds and can lessen scarring [3][4].

Instructions: {one_step_op_instruction_dict['Paraphrased']}
Query: What is a heirloom tomato?
Response: Many heirloom tomatoes are sweeter and lack a genetic mutation that gives tomatoes a uniform red color at the cost of the fruit's taste [1]. True heirlooms must be 'old' because they have been a variety for anywhere between 25 and 50 years [2].
Paraphrased Response: Unlike tomatoes that carry a genetic mutation that makes them look consistently red at the expense of their flavor, many heirloom tomatoes taste sweeter [1]. Genuine heirloom tomatoes have to be old due to being a variety over the last 25 to 50 years [2].

Instructions: {one_step_op_instruction_dict['Paraphrased']}
Query: What's the flavor of juicy fruit gum?
Response: the mixture of fruit flavors in Juicy Fruit is comprised of predominately lemon, orange, pineapple, and banana notes [1].
Paraphrased Response: Juicy Fruit flavor is a blend of mostly orange, lemon, banana, and pineapple flavors [1].
"""
    elif data_str == 'multihop':
        return f"""Instructions: {one_step_op_instruction_dict['Paraphrased']}
Query: Are the director of film Move (1970 Film) and director of film Méditerranée (1963 Film) from the same country?
Response: Move (1970 Film) was directed by Stuart Rosenberg, who was born on August 11, 1927, and passing on March 15, 2007, and was renowned for his work as a director in both American film and television [1]. On the other hand, Méditerranée was directed by Jean-Daniel Pollet who lived from 1936 to 2004 and was celebrated as a French filmmaker and scriptwriter [2].
Paraphrased Response: The film \'Move\' (1970) was directed by Stuart Rosenberg, born on August 11, 1927, and who died on March 15, 200 [1]. He was known for his distinguished career in American cinema and television [1]. Conversely, Jean-Daniel Pollet (1936-2004), a French filmmaker and screenwriter recognized for his notable work, directed \'Méditerranée\' [2]. 

Instructions: {one_step_op_instruction_dict['Paraphrased']}
Query: What nationality is the director of film Wedding Night In Paradise (1950 Film)?
Response: Wedding Night in Paradise  is a 1950 West German musical comedy film directed by Géza von Bolváry [1]. Géza von Bolváry (full name Géza Maria von Bolváry-Zahn; 26 December 1897 – 10 August 1961) was a Hungarian actor, screenwriter and film director [2].
Paraphrased Response: Géza von Bolváry directed the 1950 West German musical comedy \'Wedding Night in Paradise\' [1]. Born on December 26, 1897, Géza Maria von Bolváry-Zahn was a Hungarian talent known for his work as an actor, screenwriter, and director, and he passed away on August 10, 1961 [2].

Instructions: {one_step_op_instruction_dict['Paraphrased']}
Query: When did the performer of song A Pretty Girl Milking Her Cow die?
Response: Judy Garland (born Frances Ethel Gumm; June 10, 1922 – June 22, 1969) was an American actress, singer and dancer [1]. A Pretty Girl Milking Her Cow enjoyed a revival when an updated swing version sung by Irish-American singer/actress Judy Garland was featured in the 1940 film Little Nellie Kelly [2].
Paraphrased Response: American actress, singer, and dancer Judy Garland, originally named Frances Ethel Gumm, lived from June 10, 1922, to June 22, 1969 [1]. Notably, her rendition of \'A Pretty Girl Milking Her Cow\' experienced a resurgence in popularity when her updated swing version was showcased in the 1940 film \'Little Nellie Kelly\' [2].

Instructions: {one_step_op_instruction_dict['Paraphrased']}
Query: Do both films The Falcon (Film) and Valentin The Good have the directors from the same country?
Response: Banović Strahinja( Serbian Cyrillic:' Бановић Страхиња', released internationally as The Falcon) is a 1981 Yugoslavian- German adventure film written and directed by Vatroslav Mimica [1]. Vatroslav Mimica( born 25 June 1923) is a Croatian film director and screenwriter [2]. Valentin the Good is a 1942 Czech comedy film directed by Martin Frič and Martin Frič( 29 March 1902 – 26 August 1968) was a Czech film director, screenwriter and actor [3].
Paraphrased Response: 'Banović Strahinja' (Serbian Cyrillic: 'Бановић Страхиња', internationally known as 'The Falcon') is an adventure film from 1981, produced as a Yugoslavian-German collaboration and crafted by writer and director Vatroslav Mimica [1]. Vatroslav Mimica, born on June 25, 1923, is a screenwriter and director from Croatia [2]. Directed by Martin Frič, 'Valentin the Good' is a Czech comedy film that premiered in 1942 [3]. Martin Frič, born on March 29, 1902, and passed away on August 26, 1968, was an actor, screenwriter, and director from Czechia [3].

Instructions: {one_step_op_instruction_dict['Paraphrased']}
Query: What is the date of birth of Mina Gerhardsen's father?
Response: Mina Gerhardsen (born 14 September 1975) is a Norwegian politician for the Labour Party [1]. She is the daughter of Rune Gerhardsen and Tove Strand [2]. Rune Gerhardsen (born 13 June 1946) is a Norwegian politician [3].
Paraphrased Response: Mina Gerhardsen, born on September 14, 1975, is a politician from Norway affiliated with the Labour Party [1]. She is the daughter of Rune Gerhardsen and Tove Strand [2]. Rune Gerhardsen, born on June 13, 1946, is also a Norwegian politician [3].

Instructions: {one_step_op_instruction_dict['Paraphrased']}
Query: When is the composer of film Sruthilayalu 's birthday?
Response: Sruthilayalu is a 1987 Indian Telugu-language musical drama film, written and directed by K. Viswanath [1]. The film stars Rajasekhar and Sumalata with soundtrack composed by K. V. Mahadevan [2]. Krishnankoil Venkadachalam Mahadevan (14 March 1918 – 21 June 2001) was an Indian composer, singer-songwriter, music producer, and musician known for his works in Tamil cinema, Telugu cinema, Kannada cinema, and Malayalam cinema [3].
Paraphrased Response: 'Sruthilayalu' is a 1987 musical drama film in the Telugu language from India, crafted by writer and director K. Viswanath [1]. The movie features Rajasekhar and Sumalata in leading roles, with a soundtrack created by K. V. Mahadevan [2]. Krishnankoil Venkadachalam Mahadevan (March 14, 1918 – June 21, 2001) was an Indian singer-songwriter, musician, composer, and music producer renowned for his contributions to Telugu, Tamil, Malayalam, and Kannada cinema [3].

Instructions: {one_step_op_instruction_dict['Paraphrased']}
Query: When did Fatima Bint Mubarak Al Ketbi's husband die?
Response: Fatima bint Mubarak Al Ketbi is the third wife of Sheikh Zayed bin Sultan Al Nahyan [1]. Sheikh Zayed bin Sultan Al Nahyan ; 6 May 1918 – 2 November 2004) was the ruler of Abu Dhabi for more than 30 years (6 August 1966 – 2 November 2004) [2].
Paraphrased Response: Fatima bint Mubarak Al Ketbi is the wife of Sheikh Zayed bin Sultan Al Nahyan from his third marriage [1]. Sheikh Zayed bin Sultan Al Nahyan, born on May 6, 1918, and passing on November 2, 2004, ruled Abu Dhabi for over 30 years, from August 6, 1966, to November 2, 2004 [2].

Instructions: {one_step_op_instruction_dict['Paraphrased']}
Query: Who is the mother-in-law of Andrew Murray (Scottish Soldier)?
Response: Sir Andrew Murray married Christina Bruce, a sister of King Robert I of Scotland [1]. Christina Bruce (c. 1278 – 1356/1357), also known as Christina de Brus, was a daughter of Marjorie, Countess of Carrick, and her husband, Robert de Brus, 'jure uxoris' Earl of Carrick, as well as a sister of Robert the Bruce, King of Scots [2].
Paraphrased Response: Sir Andrew Murray took Christina Bruce, sister of King Robert I of Scotland, as his wife [1]. Born around 1278 and living until 1356/1357, Christina Bruce, also called Christina de Brus, was the daughter of Marjorie, Countess of Carrick, and Robert de Brus, Earl of Carrick by virtue of his marriage [2]. She was also the sister of Robert the Bruce, the King of Scots [2].

Instructions: {one_step_op_instruction_dict['Paraphrased']}
Query: When did the performer of song A Pretty Girl Milking Her Cow die?
Response: Judy Garland (born Frances Ethel Gumm; June 10, 1922 – June 22, 1969) was an American actress, singer and dancer [1]. A Pretty Girl Milking Her Cow enjoyed a revival when an updated swing version sung by Irish-American singer/actress Judy Garland was featured in the 1940 film Little Nellie Kelly [2].
Paraphrased Response: Frances Ethel Gumm, known professionally as Judy Garland (June 10, 1922 – June 22, 1969), was a multifaceted American performer, excelling in acting, singing, and dancing [1]. The song 'A Pretty Girl Milking Her Cow' experienced a resurgence with a modern swing rendition performed by Irish-American singer and actress Judy Garland, featured in the 1940 movie 'Little Nellie Kelly' [2]."""
    elif data_str == 'mash':
        return f"""Instructions: {one_step_op_instruction_dict['Paraphrased']}
Query: What can you do to prevent migraine with brainstem aura?
Response: To prevent migraine with brainstem aura, it helps to avoid the things that usually cause one [1]. This includes being sure to Get enough sleep [2], Limit your stress [3], and Exercise daily [4]. Additionally, Eat a balanced diet [5], Avoid drugs and alcohol [6], Don't skip meals [7],  and Limit caffeine [8]. Avoid any foods that have been triggers [9]. Furthermore, If you don't respond to other treatments and you have 4 or more migraine days a month, your doctor may suggest preventive medicines, including seizure medicines, blood pressure medicines (like beta blockers and calcium channel blockers), and some antidepressants [10]. Also, CGRP inhibitors are a new class of preventive medicine [11].
Paraphrased Response: To avert migraines with brainstem aura, steering clear of common triggers is beneficial [1]. This encompasses getting adequate sleep, minimizing stress, and engaging in daily physical activity [2][3][4]. Moreover, maintaining a nutritious diet, abstaining from drugs and alcohol, not missing meals, reducing caffeine intake, and steering clear of any known food triggers are good precautions [5][6][7][8]. Additionally, if you experience four or more days of migraines a month and haven't found relief with other treatments, your healthcare provider might recommend preventive medications [10]. Such medications may include seizure medicine, blood pressure medicines (including beta blockers and calcium channel blockers), and certain antidepressants [10]. Additionally, there is a new class of preventative medication called CGRP inhibitors [11].

Instructions: {one_step_op_instruction_dict['Paraphrased']}
Query: What should I do if my child has a biting injury?
Response: The first thing to do for any biting injury is to wash the area with soap and water [1]. Even little teeth can break the skin [2]. If the bite is bleeding and the wound appears to be deep, call your child's doctor [3]. The bite may need medical treatment, which could include antibiotics or a tetanus shot, or both [4].
Paraphrased Response: Initially, cleanse the bitten area thoroughly using soap and water, as even small teeth can penetrate the skin [1][2]. Should the bite result in bleeding and a deep injury, consult your child's physician [3]. Such injuries might necessitate medical intervention, potentially involving antibiotics, a tetanus injection, or both [4].

Instructions: {one_step_op_instruction_dict['Paraphrased']}
Query: Can I give cough and cold medicines to children under 4?
Response: Talk to your pediatrician before giving OTC cold and cough medicines [1]. These medicines should not be given to children under 4 years of age, according to the FDA and the drug makers [2].
Paraphrased Response: Consult your pediatrician before giving your child over-the-counter cough and cold medicines [1]. According to both the FDA and drug makers, children under 4 years of age should not take OTC cold and cough medicines [2].

Instructions: {one_step_op_instruction_dict['Paraphrased']}
Query: How do I treat a blister?
Response: Most blisters caused by friction or minor burns do not require a doctor's care [1]. New skin will form underneath the affected area and the fluid is simply absorbed [2]. Do not puncture a blister unless it is large, painful, or likely to be further irritated [3]. The fluid-filled blister keeps the underlying skin clean, which prevents infection and promotes healing [4]. But if you need to pop a blister or it pops by itself: Use a sterilized needle (to sterilize it, put the point or edge in a flame until it is red hot, or rinse it in alcohol) [5]. Wash your hands and the area thoroughly, then make a small hole; the fluid will drain on its own [6]. If the fluid is white or yellow, the blister may be infected and needs medical attention [7]. The new skin underneath needs this protective cover [8]. Apply an antibiotic ointment or cream [9]. Look for signs of infection to develop, including pus drainage, red or warm skin surrounding the blister, or red streaks leading away from the blister [10].
Paraphrased Response: Blisters resulting from friction or minor burns generally don't necessitate medical intervention [1]. Underneath the blister, new skin will develop, and the fluid will naturally be reabsorbed [2]. It's advisable not to burst a blister unless it's sizable, causing discomfort, or at risk of additional irritation [3]. The blister's fluid acts as a barrier, safeguarding the skin beneath from infection and aiding in the healing process [4]. However, if there's a need to drain a blister, or if it bursts on its own, ensure to use a needle that has been sterilized either by heating it until it glows red or by cleansing it with alcohol [5]. Before and after the procedure, clean your hands and the affected area thoroughly, then puncture the blister gently to let the fluid out without forcing it [6]. Should the fluid appear white or yellow, it's indicative of an infection, and medical advice should be sought [7]. Avoid peeling the skin off an opened blister as the new skin underneath requires this natural bandage for protection [8]. Applying an antibacterial ointment or cream is recommended [9]. Stay vigilant for signs of infection such as pus, reddening or warmth around the blister, or red lines emanating from it [10].
"""
    else:
        print('Paraphrased few-shot examples not found for this dataset.')
        exit()

def get_entailed_few_shot_examples(data_str):
    if data_str == 'nq':
        return f"""Instructions: {one_step_op_instruction_dict['Entailed']} 
Query: What's the life cycle of a frog?
Response: The life cycle of a frog consists of three stages: egg, larva, and adult [1]. As the frog grows, it moves through these stages in a process known as metamorphosis [2]. During the egg stage, the yolk in each egg splits into more and more cells and begins to take the form of a tadpole, the larva of a frog [3]. Within one to three weeks, the egg is ready to hatch, and a tiny tadpole breaks free [4]. In the larval stage, [t]adpoles, frogs' larvae, have rudimentary gills, a mouth, and a long tail [5]. As they grow, tadpoles feed on algae and other vegetation, and eventually, they grow back legs, then front legs too [6][7]! Soon after, their body starts to change shape, and they’re able to start eating insects [8]. Next, the tadpoles’ tails shrink away, and skin grows over their gills, as they develop lungs and eardrums [9]!
Revised Response: The stages of a frog’s life cycle are egg, larva, and adult [1]. Growing through these stages is a process called metamorphosis [2]. Cells in the yolk of a frog egg form a tadpole, or a frog larva [3]. In one to three weeks, a small tadpole hatches from the egg [4]. During the larval stage, tadpoles are equipped with simple gills, a mouth, and a lengthy tail [5]. As they develop, tadpoles consume plants and algae and then grow back legs, followed by front legs [6][7]. Shortly afterward, they gain the ability to consume insects [8]. Then, the tails of the tadpoles shrink, they lose their gills, and they grow eardrums and lungs [9].

Instructions: {one_step_op_instruction_dict['Entailed']}
Query: What are the benefits of hyaluronic acid for skin care?
Response: Hyaluronic acid helps to retain moisture [1]. It makes it easier for skin to be flexible and diminishes skin lines and wrinkles [2]. Hyaluronic acid has been shown to decrease the recovery time of wounds and can lessen scarring [3][4].
Revised Response: Hyaluronic acid helps to retain moisture [1]. It makes it easier for skin to be flexible and diminishes wrinkles [2]. Hyaluronic acid may decrease the recovery time of wounds and can lessen scarring [3][4].

Instructions: {one_step_op_instruction_dict['Entailed']}
Query: What is a heirloom tomato?
Response: Unlike tomatoes that carry a genetic mutation that makes them look consistently red at the expense of their flavor, many heirloom tomatoes taste sweeter [1]. Genuine heirloom tomatoes have to be old due to being a variety over the last 25 to 50 years [2].
Revised Response: Unlike tomatoes that carry a genetic mutation that makes them look consistently red at the expense of their flavor, many heirloom tomatoes taste sweeter [1]. Genuine heirloom tomatoes have been a variety over the last 25 to 50 years [2].

Instructions: {one_step_op_instruction_dict['Entailed']}
Query: What's the flavor of juicy fruit gum?
Response: the mixture of fruit flavors in Juicy Fruit is comprised of predominately lemon, orange, pineapple, and banana notes [1].
Revised Response: Juicy Fruit flavor is a blend of mostly orange, lemon, banana, and pineapple flavors [1]."""

    elif data_str == 'eli5_nq':
        return f"""Instructions: {one_step_op_instruction_dict['Entailed']} 
Query: What's the life cycle of a frog?
Response: The life cycle of a frog consists of three stages: egg, larva, and adult [1]. As the frog grows, it moves through these stages in a process known as metamorphosis [2]. During the egg stage, the yolk in each egg splits into more and more cells and begins to take the form of a tadpole, the larva of a frog [3]. Within one to three weeks, the egg is ready to hatch, and a tiny tadpole breaks free [4]. In the larval stage, [t]adpoles, frogs' larvae, have rudimentary gills, a mouth, and a long tail [5]. As they grow, tadpoles feed on algae and other vegetation, and eventually, they grow back legs, then front legs too [6][7]! Soon after, their body starts to change shape, and they’re able to start eating insects [8]. Next, the tadpoles’ tails shrink away, and skin grows over their gills, as they develop lungs and eardrums [9]!
Revised Response: The stages of a frog’s life cycle are egg, larva, and adult [1]. Growing through these stages is a process called metamorphosis [2]. Cells in the yolk of a frog egg form a tadpole, or a frog larva [3]. In one to three weeks, a small tadpole hatches from the egg [4]. During the larval stage, tadpoles are equipped with simple gills, a mouth, and a lengthy tail [5]. As they develop, tadpoles consume plants and algae and then grow back legs, followed by front legs [6][7]. Shortly afterward, they gain the ability to consume insects [8]. Then, the tails of the tadpoles shrink, they lose their gills, and they grow eardrums and lungs [9].

Instructions: {one_step_op_instruction_dict['Entailed']}
Query: What are the benefits of hyaluronic acid for skin care?
Response: Hyaluronic acid helps to retain moisture [1]. It makes it easier for skin to be flexible and diminishes skin lines and wrinkles [2]. Hyaluronic acid has been shown to decrease the recovery time of wounds and can lessen scarring [3][4].
Revised Response: Hyaluronic acid helps to retain moisture [1]. It makes it easier for skin to be flexible and diminishes wrinkles [2]. Hyaluronic acid may decrease the recovery time of wounds and can lessen scarring [3][4].

Instructions: {one_step_op_instruction_dict['Entailed']}
Query: What is a heirloom tomato?
Response: Unlike tomatoes that carry a genetic mutation that makes them look consistently red at the expense of their flavor, many heirloom tomatoes taste sweeter [1]. Genuine heirloom tomatoes have to be old due to being a variety over the last 25 to 50 years [2].
Revised Response: Unlike tomatoes that carry a genetic mutation that makes them look consistently red at the expense of their flavor, many heirloom tomatoes taste sweeter [1]. Genuine heirloom tomatoes have been a variety over the last 25 to 50 years [2].

Instructions: {one_step_op_instruction_dict['Entailed']}
Query: What's the flavor of juicy fruit gum?
Response: the mixture of fruit flavors in Juicy Fruit is comprised of predominately lemon, orange, pineapple, and banana notes [1].
Revised Response: Juicy Fruit flavor is a blend of mostly orange, lemon, banana, and pineapple flavors [1]."""
    elif data_str == 'multihop':
        return f"""Instructions: {one_step_op_instruction_dict['Entailed']}
Query: Are the director of film Move (1970 Film) and director of film Méditerranée (1963 Film) from the same country?
Response: Move (1970 Film) was directed by Stuart Rosenberg, who was born on August 11, 1927, and passing on March 15, 2007, and was renowned for his work as a director in both American film and television [1]. On the other hand, Méditerranée was directed by Jean-Daniel Pollet who lived from 1936 to 2004 and was celebrated as a French filmmaker and scriptwriter [2].
Revised Response: No; Stuart Rosenberg directed \'Move\' and was American while Jean-Daniel Pollet directed \'Méditerranée\' and was French [1][2].

Instructions: {one_step_op_instruction_dict['Entailed']}
Query: What nationality is the director of film Wedding Night In Paradise (1950 Film)?
Response: Wedding Night in Paradise  is a 1950 West German musical comedy film directed by Géza von Bolváry [1]. Géza von Bolváry (full name Géza Maria von Bolváry-Zahn; 26 December 1897 – 10 August 1961) was a Hungarian actor, screenwriter and film director [2].
Revised Response: Géza von Bolváry directed \'Wedding Night in Paradise\' and was Hungarian [1][2].

Instructions: {one_step_op_instruction_dict['Entailed']}
Query: When did the performer of song A Pretty Girl Milking Her Cow die?
Response: Judy Garland (born Frances Ethel Gumm; June 10, 1922 – June 22, 1969) was an American actress, singer and dancer [1]. A Pretty Girl Milking Her Cow enjoyed a revival when an updated swing version sung by Irish-American singer/actress Judy Garland was featured in the 1940 film Little Nellie Kelly [2].
Revised Response: Judy Garland, the performer of \'A Pretty Girl Milking Her Cow\' died on June 22, 1969 [1][2].

Instructions: {one_step_op_instruction_dict['Entailed']}
Query: Do both films The Falcon (Film) and Valentin The Good have the directors from the same country?
Response: Banović Strahinja( Serbian Cyrillic:' Бановић Страхиња', released internationally as The Falcon) is a 1981 Yugoslavian- German adventure film written and directed by Vatroslav Mimica [1]. Vatroslav Mimica( born 25 June 1923) is a Croatian film director and screenwriter [2]. Valentin the Good is a 1942 Czech comedy film directed by Martin Frič and Martin Frič( 29 March 1902 – 26 August 1968) was a Czech film director, screenwriter and actor [3].
Revised Response: No; the director of 'The Falcon' is Vatroslav Mimica, who is from Croatia, while 'Valentin the Good' was directed by Martin Frič, from Czechia [1][2][3].

Instructions: {one_step_op_instruction_dict['Entailed']}
Query: What is the date of birth of Mina Gerhardsen's father?
Response: Mina Gerhardsen (born 14 September 1975) is a Norwegian politician for the Labour Party [1]. She is the daughter of Rune Gerhardsen and Tove Strand [2]. Rune Gerhardsen (born 13 June 1946) is a Norwegian politician [3].
Revised Response: Mina Gerhardsen's father, Rune Gerhardsen, was born on June 13, 1946 [2][3].

Instructions: {one_step_op_instruction_dict['Entailed']}
Query: When is the composer of film Sruthilayalu 's birthday?
Response: Sruthilayalu is a 1987 Indian Telugu-language musical drama film, written and directed by K. Viswanath [1]. The film stars Rajasekhar and Sumalata with soundtrack composed by K. V. Mahadevan [2]. Krishnankoil Venkadachalam Mahadevan (14 March 1918 – 21 June 2001) was an Indian composer, singer-songwriter, music producer, and musician known for his works in Tamil cinema, Telugu cinema, Kannada cinema, and Malayalam cinema [3].
Revised Response: The composer of 'Sruthilayalu', Krishnankoil Venkadachalam Mahadevan, was born on March 14, 1918 [1][2].

Instructions: {one_step_op_instruction_dict['Entailed']}
Query: When did Fatima Bint Mubarak Al Ketbi's husband die?
Response: Fatima bint Mubarak Al Ketbi is the third wife of Sheikh Zayed bin Sultan Al Nahyan [1]. Sheikh Zayed bin Sultan Al Nahyan ; 6 May 1918 – 2 November 2004) was the ruler of Abu Dhabi for more than 30 years (6 August 1966 – 2 November 2004) [2].
Revised Response: Fatima bint Mubarak Al Ketbi's husband, Sheikh Zayed bin Sultan Al Nahyan, passed away on November 2, 2004 [1][2].

Instructions: {one_step_op_instruction_dict['Entailed']}
Query: Who is the mother-in-law of Andrew Murray (Scottish Soldier)?
Response: Sir Andrew Murray married Christina Bruce, a sister of King Robert I of Scotland [1]. Christina Bruce (c. 1278 – 1356/1357), also known as Christina de Brus, was a daughter of Marjorie, Countess of Carrick, and her husband, Robert de Brus, 'jure uxoris' Earl of Carrick, as well as a sister of Robert the Bruce, King of Scots [2].
Revised Response: Sir Andrew Murray married Christina Bruce, making her mother, Marjorie, Countess of Carrick, his mother-in-law [1][2].

Instructions: {one_step_op_instruction_dict['Entailed']}
Query: When did the performer of song A Pretty Girl Milking Her Cow die?
Response: Judy Garland (born Frances Ethel Gumm; June 10, 1922 – June 22, 1969) was an American actress, singer and dancer [1]. A Pretty Girl Milking Her Cow enjoyed a revival when an updated swing version sung by Irish-American singer/actress Judy Garland was featured in the 1940 film Little Nellie Kelly [2].
Revised Response: The song 'A Pretty Girl Milking Her Cow' became especially popular from its rendition by Judy Garland, who passed away on June 22, 1969 [1][2]."""
    elif data_str == 'mash':
        return f"""Instructions: {one_step_op_instruction_dict['Entailed']} 
Query: What can you do to prevent migraine with brainstem aura?
Response: To prevent migraine with brainstem aura, it helps to avoid the things that usually cause one [1]. This includes being sure to Get enough sleep [2], Limit your stress [3], and Exercise daily [4]. Additionally, Eat a balanced diet [5], Avoid drugs and alcohol [6], Don't skip meals [7],  and Limit caffeine [8]. Avoid any foods that have been triggers [9]. Furthermore, If you don't respond to other treatments and you have 4 or more migraine days a month, your doctor may suggest preventive medicines, including seizure medicines, blood pressure medicines (like beta blockers and calcium channel blockers), and some antidepressants [10]. Also, CGRP inhibitors are a new class of preventive medicine [11].
Revised Response: To prevent migraine with brainstem aura, avoid common triggers by getting sufficient sleep, reducing stress, exercising regularly, eating a balanced diet, avoiding drugs and alcohol, and limiting caffeine [1][2][3][4][5][6][8]. If experiencing four or more migraine days a month and unresponsive to other treatments, your doctor may recommend preventive medications [10].

Instructions: {one_step_op_instruction_dict['Entailed']} 
Query: What should I do if my child has a biting injury?
Response: The first thing to do for any biting injury is to wash the area with soap and water [1]. Even little teeth can break the skin [2]. If the bite is bleeding and the wound appears to be deep, call your child's doctor [3]. The bite may need medical treatment, which could include antibiotics or a tetanus shot, or both [4].
Revised Response: For a biting injury, first wash the area with soap and water [1]. If the bite is bleeding and deep, consult your child's doctor for possible medical treatments, including antibiotics and a tetanus shot [3][4].

Instructions: {one_step_op_instruction_dict['Entailed']} 
Query: Can I give cough and cold medicines to children under 4?
Response: Talk to your pediatrician before giving OTC cold and cough medicines [1]. These medicines should not be given to children under 4 years of age, according to the FDA and the drug makers [2].
Revised Response: The FDA and drug makers advise against giving cough and cold medicines to children under 4 [1][2].

Instructions: {one_step_op_instruction_dict['Entailed']} 
Query: How do I treat a blister?
Response: Most blisters caused by friction or minor burns do not require a doctor's care [1]. New skin will form underneath the affected area and the fluid is simply absorbed [2]. Do not puncture a blister unless it is large, painful, or likely to be further irritated [3]. The fluid-filled blister keeps the underlying skin clean, which prevents infection and promotes healing [4]. But if you need to pop a blister or it pops by itself: Use a sterilized needle (to sterilize it, put the point or edge in a flame until it is red hot, or rinse it in alcohol) [5]. Wash your hands and the area thoroughly, then make a small hole; the fluid will drain on its own [6]. If the fluid is white or yellow, the blister may be infected and needs medical attention [7]. The new skin underneath needs this protective cover [8]. Apply an antibiotic ointment or cream [9]. Look for signs of infection to develop, including pus drainage, red or warm skin surrounding the blister, or red streaks leading away from the blister [10].
Revised Response: Most blisters from friction or minor burns can heal without medical intervention, as new skin forms underneath and the fluid is naturally absorbed [1][2]. Avoid puncturing a blister unless it's painful or likely to get irritated [3]. In this case, sterilize a needle with flame or alcohol, clean your hands and the blister area, and then gently puncture the blister to let the fluid drain [5][6]. Keep the protective skin layer over the blister and apply an antibiotic ointment [8][9]. Watch for signs of infection like pus, redness, or warmth around the blister [10].
"""
    else:
        print('Entailed few-shot examples not found for this dataset.')
        exit()

def get_abstractive_few_shot_examples(data_str):
    if data_str == 'nq':
        return f"""Instructions: {one_step_op_instruction_dict['Abstractive']}
Query: What's the life cycle of a frog?
Response: The stages of a frog’s life cycle are egg, larva, and adult [1]. In a frog egg, the yolk splits into many cells which are the beginnings of a tadpole, or a frog larva [2]. In one to three weeks, a small tadpole hatches from the egg [3]. During the larval stage, tadpoles, or frogs’ larvae, are equipped with simple gills, a mouth, and a lengthy tail [4]. As they develop, tadpoles consume plants and algae and then grow back legs, followed by front legs [5]. Shortly afterward, their body begins to shift in shape and they gain the ability to consume insects [6]. Then, the tails of the tadpoles grow smaller and skin develops over their gills while they grow eardrums and lungs [7].
Revised Response: A frog matures from egg, to tadpole, to adult in a process called metamorphosis [1]. After hatching from an egg into a tadpole, it will grow back legs and then front legs, while storing nutrients in its long tail [3][5]. As the tadpole loses its gills, it gains lungs. At this point, the tadpole is considered to be a froglet, until its tail shrinks away, at which point it is a frog.

Instructions: {one_step_op_instruction_dict['Abstractive']}
Query: What are the benefits of hyaluronic acid for skin care?
Response: Hyaluronic acid helps to retain moisture [1]. It makes it easier for skin to be flexible and diminishes skin lines and wrinkles [2].
Revised Response: Hyaluronic acid helps keep skin moist and wrinkle-free [1][2]. It can also reduce scarring, which may be especially beneficial for people who have had severe acne in the past.

Instructions: {one_step_op_instruction_dict['Abstractive']}
Query: What is a heirloom tomato?
Response: Unlike tomatoes that carry a genetic mutation that makes them look consistently red at the expense of their flavor, many heirloom tomatoes taste sweeter [1]. Genuine heirloom tomatoes have to be old due to being a variety over the last 25 to 50 years [2].
Revised Response: Heirloom tomatoes are tomatoes that have a sweet flavor profile and are often other colors than red, including yellow, orange, and purple [1]. A tomato variety is considered heirloom once it’s been a variety for 25-50 years, although some insist that at least 100 years is necessary [2].

Instructions: {one_step_op_instruction_dict['Abstractive']}
Query: What's the flavor of juicy fruit gum?
Response: the mixture of fruit flavors in Juicy Fruit is comprised of predominately lemon, orange, pineapple, and banana notes [1].
Revised Response: Juicy Fruit flavor is a blend of mostly orange, lemon, banana, and pineapple flavors. Some say it tastes like coconut and jackfruit as well [1]."""

    elif data_str == 'eli5_nq':
        return f"""Instructions: {one_step_op_instruction_dict['Abstractive']}
Query: What's the life cycle of a frog?
Response: The stages of a frog’s life cycle are egg, larva, and adult [1]. In a frog egg, the yolk splits into many cells which are the beginnings of a tadpole, or a frog larva [2]. In one to three weeks, a small tadpole hatches from the egg [3]. During the larval stage, tadpoles, or frogs’ larvae, are equipped with simple gills, a mouth, and a lengthy tail [4]. As they develop, tadpoles consume plants and algae and then grow back legs, followed by front legs [5]. Shortly afterward, their body begins to shift in shape and they gain the ability to consume insects [6]. Then, the tails of the tadpoles grow smaller and skin develops over their gills while they grow eardrums and lungs [7].
Revised Response: A frog matures from egg, to tadpole, to adult in a process called metamorphosis [1]. After hatching from an egg into a tadpole, it will grow back legs and then front legs, while storing nutrients in its long tail [3][5]. As the tadpole loses its gills, it gains lungs. At this point, the tadpole is considered to be a froglet, until its tail shrinks away, at which point it is a frog.

Instructions: {one_step_op_instruction_dict['Abstractive']}
Query: What are the benefits of hyaluronic acid for skin care?
Response: Hyaluronic acid helps to retain moisture [1]. It makes it easier for skin to be flexible and diminishes skin lines and wrinkles [2].
Revised Response: Hyaluronic acid helps keep skin moist and wrinkle-free [1][2]. It can also reduce scarring, which may be especially beneficial for people who have had severe acne in the past.

Instructions: {one_step_op_instruction_dict['Abstractive']}
Query: What is a heirloom tomato?
Response: Unlike tomatoes that carry a genetic mutation that makes them look consistently red at the expense of their flavor, many heirloom tomatoes taste sweeter [1]. Genuine heirloom tomatoes have to be old due to being a variety over the last 25 to 50 years [2].
Revised Response: Heirloom tomatoes are tomatoes that have a sweet flavor profile and are often other colors than red, including yellow, orange, and purple [1]. A tomato variety is considered heirloom once it’s been a variety for 25-50 years, although some insist that at least 100 years is necessary [2].

Instructions: {one_step_op_instruction_dict['Abstractive']}
Query: What's the flavor of juicy fruit gum?
Response: the mixture of fruit flavors in Juicy Fruit is comprised of predominately lemon, orange, pineapple, and banana notes [1].
Revised Response: Juicy Fruit flavor is a blend of mostly orange, lemon, banana, and pineapple flavors. Some say it tastes like coconut and jackfruit as well [1]."""
    elif data_str == 'multihop':
        return f"""Instructions: {one_step_op_instruction_dict['Abstractive']}
Query: Are the director of film Move (1970 Film) and director of film Méditerranée (1963 Film) from the same country?
Response: Move (1970 Film) was directed by Stuart Rosenberg, who was born on August 11, 1927, and passing on March 15, 2007, and was renowned for his work as a director in both American film and television [1]. On the other hand, Méditerranée was directed by Jean-Daniel Pollet who lived from 1936 to 2004 and was celebrated as a French filmmaker and scriptwriter [2].
Revised Response: No; the director of \'Move\' was born in New York and was American while the director of \'Méditerranée\' was born in Nord and was French [1][2].

Instructions: {one_step_op_instruction_dict['Abstractive']}
Query: What nationality is the director of film Wedding Night In Paradise (1950 Film)?
Response: Wedding Night in Paradise  is a 1950 West German musical comedy film directed by Géza von Bolváry [1]. Géza von Bolváry (full name Géza Maria von Bolváry-Zahn; 26 December 1897 – 10 August 1961) was a Hungarian actor, screenwriter and film director [2].
Revised Response: The director of \'Wedding Night in Paradise\' was Hungarian, but also worked in Germany and Austria [1][2].

Instructions: {one_step_op_instruction_dict['Abstractive']}
Query: When did the performer of song A Pretty Girl Milking Her Cow die?
Response: Judy Garland (born Frances Ethel Gumm; June 10, 1922 – June 22, 1969) was an American actress, singer and dancer [1]. A Pretty Girl Milking Her Cow enjoyed a revival when an updated swing version sung by Irish-American singer/actress Judy Garland was featured in the 1940 film Little Nellie Kelly [2].
Revised Response: Judy Garland, the performer of \'A Pretty Girl Milking Her Cow\' died on June 22, 1969 at age 47 [1][2].

Instructions: {one_step_op_instruction_dict['Abstractive']}
Query: Do both films The Falcon (Film) and Valentin The Good have the directors from the same country?
Response: Banović Strahinja( Serbian Cyrillic:' Бановић Страхиња', released internationally as The Falcon) is a 1981 Yugoslavian- German adventure film written and directed by Vatroslav Mimica [1]. Vatroslav Mimica( born 25 June 1923) is a Croatian film director and screenwriter [2]. Valentin the Good is a 1942 Czech comedy film directed by Martin Frič and Martin Frič( 29 March 1902 – 26 August 1968) was a Czech film director, screenwriter and actor [3].
Revised Response: No; the director of 'The Falcon' is Vatroslav Mimica, who is from Croatia, while 'Valentin the Good' was directed by Martin Frič, from the Czech Republic [1][2][3]. However, during Frič's life, the Czech Republic did not exist; modern-day Czech Republic was first part of the Austro-Hungarian Empire and then, after World War I, was part of Czechoslovakia.

Instructions: {one_step_op_instruction_dict['Abstractive']}
Query: What is the date of birth of Mina Gerhardsen's father?
Response: Mina Gerhardsen (born 14 September 1975) is a Norwegian politician for the Labour Party [1]. She is the daughter of Rune Gerhardsen and Tove Strand. Rune Gerhardsen (born 13 June 1946) is a Norwegian politician [2].
Revised Response: Mina Gerhardsen's father, Rune Gerhardsen, was born on June 13, 1946 [1][2].

Instructions: {one_step_op_instruction_dict['Abstractive']}
Query: When is the composer of film Sruthilayalu 's birthday?
Response: Sruthilayalu is a 1987 Indian Telugu-language musical drama film, written and directed by K. Viswanath [1]. The film stars Rajasekhar and Sumalata with soundtrack composed by K. V. Mahadevan. Krishnankoil Venkadachalam Mahadevan (14 March 1918 – 21 June 2001) was an Indian composer, singer-songwriter, music producer, and musician known for his works in Tamil cinema, Telugu cinema, Kannada cinema, and Malayalam cinema [2].
Revised Response: The composer of 'Sruthilayalu', Krishnankoil Venkadachalam Mahadevan, was born on March 14, 1918 [1][2].

Instructions: {one_step_op_instruction_dict['Abstractive']}
Query: When did Fatima Bint Mubarak Al Ketbi's husband die?
Response: Fatima bint Mubarak Al Ketbi is the third wife of Sheikh Zayed bin Sultan Al Nahyan [1]. Sheikh Zayed bin Sultan Al Nahyan ; 6 May 1918 – 2 November 2004) was the ruler of Abu Dhabi for more than 30 years (6 August 1966 – 2 November 2004) [2].
Revised Response: Fatima bint Mubarak Al Ketbi's husband was Sheikh Zayed bin Sultan Al Nahyan, who passed away when he was 86, on November 2, 2004 [1][2].

Instructions: {one_step_op_instruction_dict['Abstractive']}
Query: Who is the mother-in-law of Andrew Murray (Scottish Soldier)?
Response: Sir Andrew Murray married Christina Bruce, a sister of King Robert I of Scotland [1]. Christina Bruce (c. 1278 – 1356/1357), also known as Christina de Brus, was a daughter of Marjorie, Countess of Carrick, and her husband, Robert de Brus, 'jure uxoris' Earl of Carrick, as well as a sister of Robert the Bruce, King of Scots [2].
Revised Response: Sir Andrew Murray's married Christina Bruce, making her mother, Marjorie, Countess of Carrick, his mother-in-law [1][2]. Marjorie is notable for her son, Robert the Bruce, King of Scots, who led the first War of Scottish Independence against England.

Instructions: {one_step_op_instruction_dict['Abstractive']}
Query: When did the performer of song A Pretty Girl Milking Her Cow die?
Response: Judy Garland (born Frances Ethel Gumm; June 10, 1922 – June 22, 1969) was an American actress, singer and dancer [1]. A Pretty Girl Milking Her Cow enjoyed a revival when an updated swing version sung by Irish-American singer/actress Judy Garland was featured in the 1940 film Little Nellie Kelly [2].
Revised Response: While the song 'A Pretty Girl Milking Her Cow' has been sung by many people, it became especially popular from its rendition by Judy Garland, who passed away on June 22, 1969 [1][2]."""
    elif data_str == 'mash':
        return f"""Instructions: {one_step_op_instruction_dict['Abstractive']}
Query: What can you do to prevent migraine with brainstem aura?
Response: To prevent migraine with brainstem aura, it helps to avoid the things that usually cause one [1]. This includes being sure to Get enough sleep, Limit your stress, and Exercise daily [2][3][4]. Additionally, Eat a balanced diet, Avoid drugs and alcohol, Don't skip meals,  and Limit caffeine [5][6][7][8][9]. Avoid any foods that have been triggers [10].
Revised Response: Preventing migraine with brainstem aura involves lifestyle adjustments and possibly medication for those with frequent episodes [1]. Key strategies include maintaining regular sleep patterns, managing stress, exercising regularly, eating a balanced diet, avoiding known food triggers, limiting caffeine and alcohol intake, and not skipping meals [2][3][4][5][6][7][8][9]. Furthermore, If you don't respond to other treatments and you have 4 or more migraine days a month, your doctor may suggest preventive medicines, including seizure medicines, blood pressure medicines (like beta blockers and calcium channel blockers), and some antidepressants. Also, CGRP inhibitors are a new class of preventive medicine. There is anecdotal evidence that triptans can worsen migraine symptoms, even though it is often prescribed by medical professionals.

Instructions: {one_step_op_instruction_dict['Abstractive']}
Query: What should I do if my child has a biting injury?
Response: The first thing to do for any biting injury is to wash the area with soap and water [1]. Even little teeth can break the skin [2]. If the bite is bleeding and the wound appears to be deep, call your child's doctor [3]. The bite may need medical treatment, which could include antibiotics or a tetanus shot, or both [4].
Revised Response: For a biting injury, immediately clean the wound with soap and water [1]. If it's bleeding or deep, contact your child's doctor for potential treatment, which may involve antibiotics or a tetanus shot [3][4]. If the child was bitten by an animal, it can be life-saving to seek medical treatment for a rabies vaccine within 24 hours, or at least within 72 hours. Even if more than 72 hours have elapsed since the bite, it is still not too late to get the shot. If left untreated, rabies can be fatal.

Instructions: {one_step_op_instruction_dict['Abstractive']}
Query: Can I give cough and cold medicines to children under 4?
Response: Talk to your pediatrician before giving OTC cold and cough medicines [1]. These medicines should not be given to children under 4 years of age, according to the FDA and the drug makers [2].
Revised Response: The FDA and drug makers advise against giving cough and cold medicines to children under 4 [2]. Also, never give adult medicines to children; only use medicines designed for children. Lastly, do not give aspirin to a child due to the risk of Reye's syndrome, which is a rare but serious condition.

Instructions: {one_step_op_instruction_dict['Abstractive']}
Query: How do I treat a blister?
Response: Most blisters caused by friction or minor burns do not require a doctor's care [1]. New skin will form underneath the affected area and the fluid is simply absorbed [2]. Do not puncture a blister unless it is large, painful, or likely to be further irritated [3]. The fluid-filled blister keeps the underlying skin clean, which prevents infection and promotes healing [4]. But if you need to pop a blister or it pops by itself: Use a sterilized needle (to sterilize it, put the point or edge in a flame until it is red hot, or rinse it in alcohol) [5]. Wash your hands and the area thoroughly, then make a small hole; the fluid will drain on its own [6]. If the fluid is white or yellow, the blister may be infected and needs medical attention [7]. Do not remove the skin over a broken blister [8]. The new skin underneath needs this protective cover [9]. Apply an antibiotic ointment or cream [10]. Look for signs of infection to develop, including pus drainage, red or warm skin surrounding the blister, or red streaks leading away from the blister [11].
Revised Response: For most blisters, medical care isn't necessary [1]. Allow new skin to form beneath the blister and the fluid to absorb [2]. To protect the blister from further irritation, it is advisable to cover it with moleskin, a durable cotton fabric with an adhesive backing available at most drug stores. If a blister is large, painful, or might be further irritated, and needs to be popped, sterilize a needle with heat or alcohol, clean your hands and the blister area, then gently puncture it to let the fluid drain, but don't remove the overlying skin as it protects the new skin [3][5][6][8]. If the blister's fluid isn't clear (indicating possible infection), seek medical attention [7]. Apply an antibiotic ointment, cover it with a bandage and monitor for signs of infection like pus, redness, or warmth [10][11]."""
    else:
        print('Abstractive few-shot examples not found for this dataset.')
        exit()

if __name__ == '__main__':
    print(tg_one_step_system_prompt)
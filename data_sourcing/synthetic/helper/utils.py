# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

relation_list = [
    'in',
    'on',
    'next to',
    'behind',
    'in front of',
    'over',
    'under',
    'above',
    'below',
    'besides',
]

fg_example = [
    ('coucal', "A vibrant coucal is perched on the branch of a lush green tree, surrounded by wildflowers."),
    ('bee eater', "A lively bee eater is elegantly perched on a branch, peering intently."),
    ('three-toed sloth', "A three-toed sloth is lazily hanging from a sturdy, tropical rainforest tree."),
    ('hay', "In the serene countryside, hundreds of neatly stacked hay bales lay scattered under the softly glowing golden sunset sky."),
    ('station wagon', "A shiny, red station wagon is parked under the dappled shade of a large oak tree, highlighting its spacious and family-friendly design."),
    ('zebra', "A zebra is gallantly trotting across the vast, sunlit plains of the African savannah, creating a captivating black and white spectacle."),
    ('vase', "In the well-lit living room, a beautifully designed, delicate vase stands out as the centerpiece, exuding an aura of elegance."),
    ('barber chair', "A shiny black barber chair sits invitingly in a bustling, well-lit barbershop."),
    ('carbonara', "A heaping plate of creamy carbonara pasta topped with fresh parsley sprigs."),
    ('mink', "In the midst of a dense forest with shimmering green leaves, a sleek mink gracefully navigates the underbrush, showcasing its rich, brown fur."),
    ('small white butterfly', "A small white butterfly gracefully flutters amongst vibrant, blooming summer flowers."),
    ('christmas stocking', "A vibrant red Christmas stocking is hanging delicately from a festively decorated mantelpiece."),
    ('horse-drawn vehicle', "An antique horse-drawn vehicle is stationed amidst a peaceful country landscape, its rustic wooden structure gleaming under the warm afternoon sun."),
    ('ruler measuring stick', "A manual craftsman is precisely measuring a wooden log with a ruler stick."),
    ('picket fence', "A tranquil suburban scene featuring multiple white picket fences surrounding well-maintained green lawns, punctuated by diverse, colorful flowerbeds."),
    ('suspension bridge', "Depicting a long suspension bridge, its steel cables elegantly stretching towards the sky, connecting two ends over a scenic river."),
    ('brain coral', "A vibrant brain coral stands out amidst the serene backdrop of underwater marine life."),
    ('revolver', "Multiple antique revolvers lie on a wooden table, gleaming under soft, ambient light."),
    ('slip-on shoe', "A pair of slip-on shoes, with their sleek, black leather exterior and comfortable, cushioned interior, are neatly placed on a wooden floor."),
    ('hand-held computer', "A hand-held computer, compact and portable, rests on a well-lit desk, surrounded by various technological paraphernalia and a steaming cup of coffee."),
    ('mattress', "A teddy bear lying face down on a bedspread covered mattress in front of a window."),
    ('refrigerator', "A nicely decorated kitchen with metallic refrigerator and blue counter."),
    ('ball', "Silver balls are lined up in the sand as people mill about in the background."),
    ('wheel', "The motorcycle's gleaming steering wheel, vivid red door reflected in the side mirror, and a youth passing by, creating a dynamic urban tableau."),
    ('plane', "A group of trick planes turned upside down leaving smoke trails."),
    ('vehicle', "Army vehicles, including a U.S. Army jeep and aircraft in a hangar or on display"),
    ('boy', "a little boy wearing sunglasses laying on a shelf in a basement."),
    ('fence', "a man standing near a fence as reflected in a side-view mirror of a red car."),
    ('wood table', "A footed glass with water in front of a glass with ice tea, and green serpentine bottle with pink flowers, all on a wood table in front of chair, with a window to city view."),
    ('toilet', "A black and white toilet sitting in a bathroom next to a plant filled with waste."),
    ('table lamp', "A textured brass table lamp, casting a warm, golden glow, accents a cozy reading nook beside a leather armchair and a stack of books."),
    ('hair dryer', "A modern sleek and white hair dryer, with a textured grip, stands next to a set of hairbrushes."),
    ('street sign', "The street signs indicate which way a car can and cannot turn while the signal light controls traffic."),
    ('instrument', "Man dressed in Native American clothes protecting musical instruments from the rain with an umbrella."),
    ('train', "A man and a cow's faces are near each other as a train passes by on a bridge."),
    ('giraffe', "A couple of large giraffe standing next to each other."),
    ('red admiral butterfly', "a red admiral butterfly, alights upon a dew-kissed sunflower, wings glistening under the soft morning light."),
    ('stupa', "Surrounded by verdant foliage, a white stupa rises, adorned with golden accents and intricate patterns, while devotees circle its base offering prayers."),
    ('elephant', "A group of elephants being led into the water."),
    ('bottle', "Motorcycles parked on a street with a bottle sitting on the seat of the nearest the camera."),
    ('trombone', "On a polished wooden stage, a gleaming brass trombone rests, its slide extended, next to scattered sheet music and a muted trumpet."),
    ('keyboard', "Sleek black keyboard with illuminated backlit keys, a soft wrist rest, and a nearby wireless mouse on a textured matte desk surface."),
    ('bear', "The brown bear sits watching another bear climb the rocks"),
    ('snowboard', "A man standing next to his snowboard posing for the camera."),
    ('railway', "a woman and her son walking along the tracks of a disused railway."),
    ('sand', "the waves and the sand on the beach close up"),
    ('pixel', "very colorful series of squares or pixels in all the colors of the spectrum , from light to dark"),
    ('cigar', "a burning cigar in a glass ashtray with a blurred background."),
    ('music', "happy girl listening music on headphones and using tablet in the outdoor cafe."),
    ('earring', "this gorgeous pair of earrings were featured in april issue."),
    ('cliff', "Steep cliff, jagged edges against azure sky, with seabirds soaring and waves crashing below."),
    ('corn cob', "Fresh corn cob, golden kernels glistening with dew, nestled amid green husks in a sunlit field."),
]


bg_example = [
    ('archaeological excavation',
     "In this intriguing scene, archaeologists meticulously uncover ancient relics at an archaeological excavation site filled with historical secrets and enigmas."),
    ('formal garden',
     "This is an immaculately kept formal garden, with perfectly trimmed hedges, colorful, well-arranged flower beds, and classic statuary, giving a vibe of tranquil sophistication."),
    ('veterinarians office',
     "The busy veterinarian's office is a hive of activity with pets awaiting treatment and care."),
    ('elevator', "A modern, well-lit elevator interior with shiny metal walls and sleek buttons."),
    ('heliport',
     "Situated in a lively area, the heliport stands out with numerous helicopters taking off and landing against the city's skyline."),
    ('airport terminal',
     "In the spacious airport terminal, travelers hurriedly navigate through check-ins and security, making it a hive of constant activity."),
    ('car interior',
     "Inside the car, the leather seats exude luxury, contrasted by the high-tech dashboard, creating an atmosphere of sleek comfort and convenience."),
    ('train interior', "The inside of the train offers a spacious setting with numerous comfortable seats."),
    ('candy store',
     "The sweet aroma of sugared treats fills the air in a vibrant candy store, adorned with colourful candies and cheerful customers."),
    ('bus station',
     "The bustling bus station thrums with restless energy, as travelers navigate through the crowded space, awaiting their journeys amid the echoes of departing buses."),
    ('castle',
     "Nestled amidst towering mountains, the majestic castle spews ancient grandeur, with its stone walls and towering turrets exuding tranquility and timeless mystique."),
    ('palace',
     "The grand palace exudes regality, radiant under the sun, showcasing ornate decorations, intricate sculptures, and exquisite architectural sophistication."),
    ('kitchen',
     "The heart of the home unfolds in the kitchen, characterized by stainless steel appliances, navy blue cabinets, and a patterned tile backsplash."),
    ('raceway',
     "The high-speed adrenaline-filled atmosphere of the raceway is pulsing with the roars of powerful engines and excited cheering fans."),
    ('bakery',
     "The warm, inviting bakery is filled with the intoxicating aroma of fresh bread, assorted pastries, and brewing coffee."),
    ('medina',
     "This ancient, labyrinth-like medina exudes an air of mystique with its vibrantly decorated shops lining narrow, stone-cobbled pathways."),
    ('skyscraper',
     "The city skyline is dominated by towering skyscrapers, creating a captivating blend of technology and architectural innovation."),
    ('supermarket',
     "The supermarket scene is lively, filled with individuals scanning shelves, children reaching for treats, and clerks restocking fresh produce."),
    ('closet', "The compact closet, brimming with clothes and shoes, exudes a feeling of organization."),
    ('assembly line',
     "In the heart of a busy factory, an orderly assembly line hums with continuous activity, filled with workers focused on their precision tasks."),
    ('palace room',
     "A man in military dress uniform stands in an ornate palace room with antique furniture and Christmas decorations."),
    ('barn doorway', "A farmer holding an animal back while another farmer stands in a barn doorway."),
    ('food court',
     "A bustling food court with a variety of culinary stalls, featuring vibrant signage, aromatic dishes, and communal seating, creates a diverse dining experience."),
    ('mountain',
     "Majestic mountains, their peaks dusted with snow, overlook a serene alpine lake where hikers and photographers gather to enjoy the breathtaking scenery."),
    ('squash court',
     "Against a clear glass wall, a squash court with gleaming wooden floors, white boundary lines, and two rackets awaits players."),
    ('subway station', "Dimly lit subway station with graffiti-covered walls, commuters waiting"),
    ('restaurant',
     "Cozy restaurant with wooden tables, ambient lighting, patrons chatting, and plates filled with colorful dishes, framed by exposed brick walls and hanging green plants."),
    ('field', "there is a large heard of cows and a man standing on a field."),
    ('aquarium',
     "Amidst vivid coral formations, an aquarium teems with colorful fish, shimmering under soft blue lights."),
    ('market', "A large group of bananas on a table outside in the market."),
    ('park', "a young boy is skating on ramps at a park"),
    ('beach', "old fishing boats beached on a coastal beach in countryside."),
    ('grass', "little boy sitting on the grass with drone and remote controller."),
]


texture_example = [
    ('woven', "The woven basket's intricate pattern creates a visually captivating and tactile surface."),
    ('knitted', "The knitted blanket envelops with cozy warmth"),
    ('flecked', "The stone surface was flecked, giving it a uniquely speckled and rough appearance."),
    ('bubbly', "The liquid gleamed, showcasing its bubbly, effervescent texture vividly."),
    ('cobwebbed', "The dusty corner was cobwebbed, displaying years of untouched, eerie beauty."),
    ('stained', "A weather-worn wall manifests an intriguing pattern of stained texture."),
    ('scaly', "The image showcases a close-up of a lizard's scaly, rough texture."),
    ('meshed', "A patterned image depicting the intricate, tightly-knit texture of meshed fabric."),
    ('waffled', "A fresh, golden-brown waffle displays its distinct crisply waffled texture invitingly."),
    ('pitted', "The image portrays an intriguing terrain, characterized by a pitted, moon-like surface."),
    ('studded', "A studded leather jacket gleams, highlighting its rough, tactile texture."),
    ('crystalline', "The picture showcases an exquisite, crystalline texture with stunning brilliance and clarity."),
    ('gauzy', "A delicate veil of gauzy texture enhances the ethereal, dreamy atmosphere."),
    ('zigzagged', "The photo captures the zigzagged texture, emphasizing the rhythmic, sharp-edged patterns."),
    ('pleated', "A flowing skirt delicately showcasing the intricate detail of pleated texture."),
    ('veined', "A detailed image showcasing the intricate, veined texture of a leaf."),
    ('spiralled', "The spiralled texture of the seashell creates a captivating, tactile pattern."),
    ('lacelike', "The delicate veil features an intricate, lacelike texture, exuding elegant sophistication."),
    ('smeared', "A wall coated with thick, smeared paint exudes a rough texture."),
    ('crosshatched', "A worn, vintage book cover, richly crosshatched, exuding old-world charm."),
    ('particle', "abstract background of a heart made up of particles.")
]


fgbg_example = [
    ('stick insect', 'undergrowth', "A stick insect, masterfully camouflaged, clings to a fern amidst the sprawling, dense undergrowth of a lush, tropical forest."),
    ('black swan', 'public garden', "In the peaceful ambiance of a lush public garden, a majestic black swan gracefully glides across a shimmering emerald-green pond."),
    ('st. bernard', 'family-photo', "In the heartwarming family photo, a gregarious St. Bernard dog is seen joyfully nestled among his adoring human companions."),
    ('measuring cup', 'food prep area', "In the food prep area, multiple transparent measuring cups are neatly organized on the marble countertop."),
    ('can opener', 'hotel room', "A sleek, stainless steel can opener is sitting on the glossy dark-wood kitchenette counter of a modern, well-appointed hotel room."),
    ('small white butterfly', 'pond side', "A delicate, small white butterfly flutters gracefully above the tranquil pond side, creating a serene image amidst lush greenery."),
    ('hair dryer', 'theatre', "A sleek, professional hair dryer is positioned center stage amidst the dramatic velvet curtains and ornate details of a bustling theatre."),
    ('water bottle', 'airport', "A reusable water bottle sits on the glossy surface of a bustling airport terminal counter, amidst a backdrop of hurried travelers and departure screens."),
    ('leonberger', 'horse ranch', "Several Leonbergers are joyfully romping around a bustling horse ranch."),
    ('lighter', 'motorhome', "In the cozy, cluttered environment of a well-traveled motorhome, a sleek silver lighter holds dominion on the rustic wooden table."),
    ('slug', 'foliage', "A solitary, glistening slug meanders slowly amidst lush, dense green foliage, leaving a slimy trail on dewy leaves in its path."),
    ('ring binder', 'education department', "The ring binder, filled with important documents, sits prominently on a well-organized desk in the bustling education department."),
    ('weimaraner', 'pet store', "A sleek, silver-gray Weimaraner is spotted curiously sniffing around various pet supplies in a well-stocked and vibrant pet store."),
    ('norfolk terrier', 'countryside', "A lively Norfolk terrier joyfully bounds across a lush, green countryside, its red fur contrasting vividly with the vast open surroundings."),
    ('dalmatian', 'apple orchard', "A lively Dalmatian is playfully darting amongst the lush rows of a bountiful apple orchard, its spots contrasting against the ruby fruits."),
    ('television', 'mountain lodge', "A sleek, modern television sits prominently against the rustic, wooden walls of an inviting mountain lodge, surrounded by pine-furnished decor."),
    ('guillotine', 'horror story', "In the shadowy landscape of a suspenseful horror story, a grim, menacing guillotine looms ominously, exuding a petrifying sense of imminent dread."),
    ('hot tub', 'condominium', "A luxurious hot tub is nestled in the private balcony of a high-rise condominium, boasting spectacular cityscape views."),
    ('leaf beetle', 'plant nurseries', "A vibrant leaf beetle is diligently navigating through a lush plant nursery, its metallic sheen contrasting against the abundant green foliage."),
    ('carolina anole', 'hiking trails', "A small Carolina Anole lizard basks in the warm sunlight, gracefully draped over a gnarled tree root next to a bustling hiking trail."),
    ('girl', 'laboratory', "teenage girl and boy working in a laboratory on an experiment."),
    ('tiger', 'forest', "Two tigers are running together in the forest."),
    ('sunset', 'lake', "Golden sunset hues reflect on a calm lake, silhouetting a lone canoeist against a backdrop of fiery clouds."),
    ('building', 'mountain', "town of skyline over roofs of historic buildings with the mountains in the background."),
    ('block plane', 'weathered wood', "A block plane, its sharp blade gleaming, rests on weathered wood"),
    ('olive tree', 'soil', "single olive tree planted in the center of a dry and cracked soil"),
    ('hamster', 'pet store', "A curious hamster peers out, with pet store shelves stacked with supplies behind."),
    ('bag', 'factory', "plastic bags production line in a factory."),
    ('restaurant', 'ocean', "young pretty couple dining in a romantic atmosphere at restaurant on the boat with ocean on the background"),
    ('helicopter', 'burning forest', "a helicopter flies over a portion of burning forest."),
    ('pipe organ', 'commemoration event', "striking pipe organ dominates with its notes resonating, while a somber commemoration event unfolds in the backdrop"),
    ('rotisserie', 'wedding reception', "Rotisserie turning golden meats, with a bustling wedding reception, twinkling lights, and guests mingling."),
    ('duck', 'taiga', "A group of ducks paddle on a tranquil pond, dense taiga and towering conifers looming in the background."),
    ('tiger beetle', 'rice fields', "Amidst verdant rice fields, a shimmering tiger beetle perches prominently on a dew-kissed blade of grass."),
    ('girl', 'barn', "slow motion clip of a girl walking with her horse through a barn"),
    ('headmaster', 'graduation ceremony', "the headmaster addresses the graduating seniors during graduation ceremonies."),
    ('businessperson', 'music festival', "businessperson and guest attend music festival."),
    ('fountain', 'park', "Water cascades from an ornate fountain, surrounded by autumn-hued trees in a serene park."),
    ('speedboat', 'water', "A sleek speedboat glides on shimmering waters, powered by twin high-horsepower outboard motors."),
    ('pipe', 'beach', "a rusty water pipe on the beach."),
    ('pretzel', 'home kitchen', "Golden pretzel rests on a wooden board, with a cozy home kitchen, pots and tiled backsplash, behind."),
    ('forklift', 'paper mill', "A forklift transports hefty paper rolls amidst the industrial bustling paper mill."),
    ('lotion', 'therapy center', "Blue lotion bottles lined up at a thalasso therapy center by the ocean."),
    ('guinea pig', 'sand dunes', "Guinea pig exploring vast golden sand dunes, with tiny footprints trailing behind."),
    ('groom', 'wedding ceremony', "father of groom congratulating him after the wedding ceremony."),
    ('fishing boat', 'village', "fishing boats moored at fishing village a suburb of capital of the state,"),
    ('red fox', 'yard', "wild red fox sitting on a partially snow covered front yard of a house in the suburbs of a small city"),
    ('grey wolf', 'woodland areas', "A grey wolf prowls silently, eyes alert, through dense, misty woodland areas with moss-covered trees."),
    ('cheetah', 'edges of swamplands', "A cheetah crouches, poised and watchful, at the lush edges of murky swamplands."),
    ('wine bottle', 'living room', "in the living room, a person si opening a wine bottle with corkscrew with wooden barrel"),
]


fgrel_example = [
    ('product packet / packaging', 'next to', "A vibrant product packet, adorned with colorful labels and intricate designs, is neatly placed next to an elegant crystal glass."),
    ('croquet ball', 'behind', "A vivid, red croquet ball rests serenely, hiding behind a worn, rustic wooden fence in a sun-kissed, lush green lawn."),
    ('bassoon', 'in front of', "A beautifully crafted bassoon stands elegantly in front of a backdrop of velvet curtains, ready to perform at a concert."),
    ('grand piano', 'above', "A gorgeous, antique chandelier is suspended above the glossy black grand piano, illuminating it with warm, opulent light."),
    ('bolo tie', 'behind', "A beautifully crafted bolo tie is casually hung, indicating its previous use, behind a rustic, well-polished wooden shelf."),
    ('waffle iron', 'next to', "A large, black waffle iron is placed next to a sparkling glass jar filled with golden maple syrup on a wooden countertop."),
    ('komodo dragon', 'below', "A young child grins excitedly, peering down from a secure bridge, as a colossal Komodo dragon sprawls lazily below in the wildlife park."),
    ('vaulted or arched ceiling', 'besides', "Besides the grand marble statue, glimpses of an intricate vaulted or arched ceiling add to the room’s majestic charm."),
    ('gossamer-winged butterfly', 'next to', "A lovely, vibrant gossamer-winged butterfly is gently perched next to a dew-kissed red rose in an early morning garden."),
    ('kit fox', 'in front of', "A group of small, fluffy, golden kit foxes is playfully gathered in front of a lush, green, towering forest backdrop."),
    ('koala', 'in', "A cute, fuzzy koala is visibly relaxed, nestled contentedly in the crook of a towering, lush green eucalyptus tree."),
    ('centipede', 'above', "A vibrant green centipede is effortlessly crawling on a tree branch, positioned distinctly above a patch of untouched fern leaves."),
    ('mountain bike', 'above', "A mountain bike is displayed prominently above the rustic mantlepiece, showcasing its sleek design and intricate details."),
    ('wallaby', 'above', "A fluffy, brown wallaby is leaping high, appearing as if it is effortlessly floating above a lush, green Australian field."),
    ('giant panda', 'on', "A playful giant panda is perched on a sturdy tree branch, munching on fresh green bamboo amidst the tranquil forest ambiance."),
    ('beagle', 'on', "A pack of adorable beagles are spotted lounging on an expansive, sunbathed meadow with colorful wildflowers sprouting around them."),
    ('beach', 'on', "A vivid sunset is on display over a sprawling beach, casting warm hues on the waves gently lapping at the sandy shore."),
    ('grey whale', 'on', "A voluminous grey whale is majestically breaching, its massive body on display against the azure backdrop of the expansive ocean."),
    ('tractor', 'in front of', "A bright red tractor is parked in front of a rustic, weathered barn, casting long shadows under the golden afternoon sun."),
    ('cabbage', 'besides', "A vibrant image portrays a lush, green cabbage, glistening with dewdrops, nestled besides a rustic, wooden crate full of freshly harvested vegetables."),
]


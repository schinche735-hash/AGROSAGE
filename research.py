"""AI-assisted disease guidance with offline fallback; AI never chooses the diagnosis."""
import json, os, re, sqlite3, urllib.error, urllib.request
from pathlib import Path

LANG = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
CACHE_DB = Path(__file__).resolve().parent / "data" / "disease_knowledge.db"
CACHE_VERSION = 2

def _cache_get(crop, disease, lang):
    try:
        with sqlite3.connect(CACHE_DB) as con:
            row = con.execute(
                "SELECT payload FROM disease_knowledge WHERE crop=? AND disease=? AND lang=? AND version=?",
                (crop.strip().lower(), disease.strip().lower(), lang, CACHE_VERSION)
            ).fetchone()
        return json.loads(row[0]) if row else None
    except (sqlite3.Error, ValueError, OSError):
        return None

def _cache_put(crop, disease, lang, payload):
    try:
        CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(CACHE_DB) as con:
            con.execute("""CREATE TABLE IF NOT EXISTS disease_knowledge(
                crop TEXT NOT NULL, disease TEXT NOT NULL, lang TEXT NOT NULL,
                version INTEGER NOT NULL, payload TEXT NOT NULL, source TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(crop,disease,lang))""")
            con.execute("""INSERT INTO disease_knowledge(crop,disease,lang,version,payload,source,updated_at)
                VALUES(?,?,?,?,?,?,CURRENT_TIMESTAMP)
                ON CONFLICT(crop,disease,lang) DO UPDATE SET
                version=excluded.version,payload=excluded.payload,source=excluded.source,updated_at=CURRENT_TIMESTAMP""",
                (crop.strip().lower(), disease.strip().lower(), lang, CACHE_VERSION,
                 json.dumps(payload, ensure_ascii=False), payload.get("status", "offline")))
    except sqlite3.Error:
        # A cache write failure must not make disease prediction fail.
        pass

# Short, conservative offline guidance for the ten supported disease classes.
# It is shown when Gemini is unavailable and is not a substitute for local diagnosis.
GUIDANCE = {
    "common rust": {
        "en": ("Common rust is a fungal disease of maize. It commonly causes raised, reddish-brown to cinnamon pustules on leaves; severe infection can reduce green leaf area and grain filling.", "Use resistant hybrids where available, remove volunteer maize and crop residues where practical, avoid repeatedly growing maize in the same field, and monitor leaves early. Ask a local KVK/agriculture officer before any chemical treatment."),
        "hi": ("कॉमन रस्ट मक्का का एक फफूंदजनित रोग है। पत्तियों पर लाल-भूरे या दालचीनी रंग के उभरे छोटे फफोले दिखाई दे सकते हैं। अधिक संक्रमण से हरी पत्ती का क्षेत्र और दाने भरने की क्षमता घट सकती है।", "उपलब्ध होने पर रोग-प्रतिरोधी संकर चुनें, स्वयं उगे मक्के और संक्रमित अवशेषों का उचित प्रबंधन करें, लगातार मक्का उगाने से बचें और पत्तियों की नियमित जाँच करें। किसी भी रासायनिक उपचार से पहले स्थानीय KVK या कृषि अधिकारी से सलाह लें।"),
        "mr": ("कॉमन रस्ट हा मका पिकावरील बुरशीजन्य रोग आहे. पानांवर लालसर-तपकिरी किंवा दालचिनी रंगाचे उठावदार फोड दिसू शकतात. तीव्र प्रादुर्भावामुळे हिरव्या पानांचे क्षेत्र आणि दाणे भरण्यावर परिणाम होऊ शकतो.", "उपलब्ध असल्यास रोगप्रतिकारक संकरित वाण निवडा, आपोआप उगवलेला मका व रोगट अवशेषांचे योग्य व्यवस्थापन करा, सतत मका घेणे टाळा आणि पानांची नियमित तपासणी करा. रासायनिक उपचारापूर्वी स्थानिक KVK किंवा कृषी अधिकाऱ्यांचा सल्ला घ्या.")
    },
    "northern leaf blight": {
        "en": ("Northern leaf blight is a fungal disease of maize that often produces long, cigar-shaped grey-green to tan lesions on leaves. Lesions may merge and dry large areas of foliage in humid conditions.", "Use locally recommended resistant hybrids, rotate with non-host crops, manage infected crop residue where practical, and avoid prolonged leaf wetness where irrigation can be managed. Confirm diagnosis with a KVK before treatment."),
        "hi": ("नॉर्दर्न लीफ ब्लाइट मक्का का फफूंदजनित रोग है। पत्तियों पर लंबे, सिगार जैसे धूसर-हरे से भूरे धब्बे बन सकते हैं; नमी में धब्बे मिलकर पत्तियों का बड़ा भाग सुखा सकते हैं।", "स्थानीय रूप से अनुशंसित रोग-प्रतिरोधी संकर अपनाएँ, गैर-मेज़बान फसलों के साथ फसल चक्र रखें और संक्रमित अवशेषों का प्रबंधन करें। उपचार से पहले KVK से रोग की पुष्टि कराएँ।"),
        "mr": ("नॉर्दर्न लीफ ब्लाइट हा मक्यावरील बुरशीजन्य रोग आहे. पानांवर लांब, सिगारसारखे राखाडी-हिरवे ते तपकिरी डाग दिसू शकतात. दमट हवेत डाग एकत्र येऊन पानांचा मोठा भाग वाळवू शकतात.", "स्थानिक शिफारशीनुसार रोगप्रतिकारक संकरित वाण वापरा, बिगर-यजमान पिकांसोबत फेरपालट करा आणि रोगट अवशेषांचे व्यवस्थापन करा. उपचारापूर्वी KVK कडून रोगाची खात्री करून घ्या.")
    },
    "early blight": {
        "en": ("Early blight is commonly associated with Alternaria fungi. Older leaves often develop brown spots with concentric, target-like rings and yellowing around them. Other stresses can look similar, so the image prediction needs confirmation.", "Remove badly affected lower leaves when practical, avoid wetting foliage, provide adequate spacing and balanced nutrition, rotate crops, and manage volunteer plants and residues. Confirm the disease locally before using any pesticide."),
        "hi": ("अर्ली ब्लाइट प्रायः अल्टरनेरिया फफूंद से जुड़ा रोग है। पुरानी पत्तियों पर गोल भूरे धब्बे और उनमें लक्ष्य जैसे गोल छल्ले बन सकते हैं; आसपास पीलापन भी दिख सकता है। अन्य समस्याएँ भी ऐसी दिख सकती हैं, इसलिए पहचान की पुष्टि जरूरी है।", "बहुत प्रभावित निचली पत्तियों को जहाँ संभव हो हटाएँ, पत्तियों को लंबे समय तक गीला रहने से बचाएँ, उचित दूरी और संतुलित पोषण रखें तथा फसल चक्र अपनाएँ। किसी भी कीटनाशक/फफूंदनाशक से पहले स्थानीय सलाह लें।"),
        "mr": ("अर्ली ब्लाइट हा बहुधा अल्टरनेरिया बुरशीशी संबंधित रोग आहे. जुन्या पानांवर एकमेकांत वर्तुळे असलेले तपकिरी डाग आणि भोवती पिवळेपणा दिसू शकतो. इतर कारणांमुळेही अशी लक्षणे दिसू शकतात; त्यामुळे निदानाची खात्री आवश्यक आहे.", "शक्य असल्यास जास्त बाधित खालची पाने काढा, पाने दीर्घकाळ ओली राहू देऊ नका, योग्य अंतर व संतुलित पोषण ठेवा आणि पीक फेरपालट करा. कोणतेही बुरशीनाशक वापरण्यापूर्वी स्थानिक तज्ज्ञांचा सल्ला घ्या.")
    },
    "late blight": {
        "en": ("Late blight is a potentially fast-spreading disease caused by the oomycete Phytophthora infestans. Potato leaves may develop water-soaked dark patches, sometimes with pale growth at lesion edges in humid weather; stems and tubers can also be affected.", "Inspect crops frequently, use healthy certified seed and locally recommended varieties, avoid moving infected plants/tubers, and manage volunteer potatoes and cull piles. Seek prompt KVK advice because outbreaks can spread quickly; use only locally registered products according to label directions."),
        "hi": ("लेट ब्लाइट फाइटोफ्थोरा इन्फेस्टन्स नामक जीव के कारण होने वाला तेजी से फैल सकने वाला रोग है। आलू की पत्तियों पर पानी सोखे हुए जैसे गहरे धब्बे बन सकते हैं; नम मौसम में धब्बों के किनारों पर हल्की सफेद वृद्धि दिख सकती है। तने और कंद भी प्रभावित हो सकते हैं।", "फसल की बार-बार जाँच करें, स्वस्थ प्रमाणित बीज-कंद और स्थानीय अनुशंसित किस्में अपनाएँ, संक्रमित पौधों/कंदों को न फैलाएँ और स्वयं उगे आलू हटाएँ। रोग तेजी से फैल सकता है, इसलिए शीघ्र KVK से सलाह लें; केवल पंजीकृत उत्पाद लेबल के अनुसार ही इस्तेमाल करें।"),
        "mr": ("लेट ब्लाइट हा Phytophthora infestans मुळे होणारा आणि वेगाने पसरू शकणारा रोग आहे. बटाट्याच्या पानांवर पाणचट दिसणारे काळसर डाग पडू शकतात; दमट हवेत डागांच्या कडेला फिकट वाढ दिसू शकते. खोडे व कंदही बाधित होऊ शकतात.", "पिकाची वारंवार पाहणी करा, निरोगी प्रमाणित बियाणे-कंद व स्थानिक शिफारस केलेले वाण वापरा आणि बाधित झाडे/कंद इतरत्र नेऊ नका. रोग वेगाने पसरू शकतो; त्वरित KVK चा सल्ला घ्या आणि फक्त नोंदणीकृत उत्पादन लेबलनुसार वापरा.")
    },
    "bacterial spot": {
        "en": ("Bacterial spot of tomato causes small dark, sometimes water-soaked leaf spots that may develop yellow halos; fruit can develop raised or scabby spots. Visual symptoms can overlap with other leaf diseases.", "Use clean seed and healthy transplants, avoid working among wet plants, reduce overhead irrigation, sanitize tools, rotate away from tomato and pepper, and remove severely infected material safely. Ask a local expert to confirm whether the problem is bacterial before treatment."),
        "hi": ("टमाटर में बैक्टीरियल स्पॉट से पत्तियों पर छोटे गहरे, कभी-कभी पानी सोखे हुए धब्बे बनते हैं और उनके चारों ओर पीलापन हो सकता है। फलों पर उभरे या खुरदरे धब्बे दिख सकते हैं।", "स्वच्छ बीज और स्वस्थ पौध लगाएँ, गीले पौधों में काम करने से बचें, ऊपर से सिंचाई कम करें, औजार साफ रखें और टमाटर/मिर्च से फसल चक्र अपनाएँ। उपचार से पहले स्थानीय विशेषज्ञ से पुष्टि कराएँ।"),
        "mr": ("टोमॅटोवरील बॅक्टेरियल स्पॉटमध्ये पानांवर लहान काळसर, कधी पाणचट डाग दिसतात आणि भोवती पिवळसर कडा असू शकतात. फळांवर उंचवट्यासारखे किंवा खरबरीत डाग पडू शकतात.", "स्वच्छ बियाणे व निरोगी रोपे वापरा, ओली झाडे हाताळणे टाळा, वरून पाणी देणे कमी करा, अवजारे स्वच्छ ठेवा आणि टोमॅटो/मिरचीपासून पीक फेरपालट करा. उपचारापूर्वी स्थानिक तज्ज्ञांकडून खात्री करून घ्या.")
    },
    "leaf mold": {
        "en": ("Tomato leaf mold is a fungal disease favored by high humidity and poor air movement. Pale yellow patches may appear on the upper leaf surface, with olive-green to brown velvety growth underneath.", "Improve spacing and ventilation, avoid wetting leaves, water near the soil, remove infected lower leaves and crop debris, and use locally recommended resistant varieties. Confirm the diagnosis before treatment."),
        "hi": ("टमाटर लीफ मोल अधिक नमी और कम हवा के आवागमन में बढ़ने वाला फफूंदजनित रोग है। पत्ती की ऊपरी सतह पर हल्के पीले धब्बे और नीचे जैतूनी-हरे से भूरे मखमली जमाव दिख सकते हैं।", "पौधों के बीच उचित दूरी और हवा का आवागमन रखें, पत्तियों को गीला करने से बचें, मिट्टी के पास सिंचाई करें और संक्रमित पत्तियाँ/अवशेष हटाएँ। उपचार से पहले रोग की पुष्टि कराएँ।"),
        "mr": ("टोमॅटो लीफ मोल्ड हा जास्त आर्द्रता व कमी हवा खेळती राहणाऱ्या परिस्थितीत वाढणारा बुरशीजन्य रोग आहे. पानांच्या वरच्या बाजूला फिकट पिवळे डाग आणि खालच्या बाजूला ऑलिव्ह-हिरवा ते तपकिरी मखमली थर दिसू शकतो.", "झाडांमध्ये योग्य अंतर व हवा खेळती ठेवा, पानांवर पाणी पडणे टाळा, मुळाजवळ पाणी द्या आणि बाधित पाने/अवशेष काढा. उपचारापूर्वी रोगाची खात्री करून घ्या.")
    },
    "septoria leaf spot": {
        "en": ("Septoria leaf spot is a fungal disease of tomato. Numerous small circular spots with pale or grey centers and dark margins can appear first on older lower leaves, often followed by yellowing and leaf drop.", "Remove infected lower leaves without spreading them through the crop, keep foliage as dry as practical, mulch to reduce soil splash, rotate crops and clean up residues. Get local confirmation before chemical control."),
        "hi": ("सेप्टोरिया लीफ स्पॉट टमाटर का फफूंदजनित रोग है। पहले निचली पुरानी पत्तियों पर छोटे गोल धब्बे बन सकते हैं जिनका बीच हल्का/धूसर और किनारा गहरा होता है; बाद में पीलापन और पत्तियाँ गिर सकती हैं।", "संक्रमित निचली पत्तियाँ सावधानी से हटाएँ, मिट्टी के छींटे कम करने के लिए मल्च का उपयोग करें, पत्तियों को सूखा रखें, फसल चक्र अपनाएँ और अवशेष साफ करें। रासायनिक नियंत्रण से पहले स्थानीय पुष्टि लें।"),
        "mr": ("सेप्टोरिया लीफ स्पॉट हा टोमॅटोवरील बुरशीजन्य रोग आहे. जुन्या खालच्या पानांवर लहान गोल डाग दिसतात; डागांचा मध्यभाग फिकट/राखाडी आणि कडा गडद असू शकतात. पुढे पाने पिवळी पडून गळू शकतात.", "बाधित खालची पाने काळजीपूर्वक काढा, मातीचे शिंतोडे कमी करण्यासाठी आच्छादन वापरा, पाने शक्यतो कोरडी ठेवा, पीक फेरपालट करा आणि अवशेष स्वच्छ करा. रासायनिक नियंत्रणापूर्वी स्थानिक खात्री घ्या.")
    },
    "tomato yellow leaf curl virus": {
        "en": ("Tomato yellow leaf curl is a viral disease commonly spread by whiteflies. Plants may be stunted, with upward-curling, yellowing small leaves and poor fruit set. Symptoms alone cannot confirm the virus.", "Use healthy seedlings, remove severely symptomatic plants where advised, control weeds that host whiteflies, use suitable insect-proof nursery practices and monitor whiteflies. Ask a KVK about locally suitable integrated whitefly management; fungicides do not cure viral disease."),
        "hi": ("टमाटर येलो लीफ कर्ल एक विषाणुजनित रोग है जो अक्सर सफेद मक्खी से फैलता है। पौधे छोटे रह सकते हैं, पत्तियाँ ऊपर की ओर मुड़कर पीली और छोटी हो सकती हैं तथा फल कम लग सकते हैं। केवल लक्षणों से पुष्टि नहीं होती।", "स्वस्थ पौध लगाएँ, सलाह के अनुसार बहुत प्रभावित पौधे हटाएँ, सफेद मक्खी के आश्रय देने वाले खरपतवार नियंत्रित करें और नर्सरी में उपयुक्त जाली/सुरक्षा अपनाएँ। सफेद मक्खी के समेकित प्रबंधन के लिए KVK से सलाह लें; फफूंदनाशक विषाणु को ठीक नहीं करते।"),
        "mr": ("टोमॅटो यलो लीफ कर्ल हा विषाणूजन्य रोग असून तो सामान्यतः पांढऱ्या माशीमुळे पसरतो. झाडांची वाढ खुंटू शकते, पाने वरच्या बाजूला वळून पिवळी व लहान होऊ शकतात आणि फळधारणा कमी होऊ शकते. केवळ लक्षणांवरून खात्री होत नाही.", "निरोगी रोपे वापरा, सल्ल्यानुसार अतिबाधित झाडे काढा, पांढऱ्या माशीला आश्रय देणारी तणे नियंत्रित करा आणि रोपवाटिकेत योग्य जाळी/संरक्षण वापरा. पांढऱ्या माशीच्या एकात्मिक व्यवस्थापनासाठी KVK चा सल्ला घ्या; बुरशीनाशके विषाणू बरे करत नाहीत.")
    },
    "healthy": {
        "en": ("The classifier did not find a familiar disease pattern for this image. This does not prove the plant is disease-free; symptoms may be early, unclear, or outside the model's supported classes.", "Keep monitoring new growth, use balanced irrigation and nutrition, maintain field hygiene, and compare symptoms with trusted local agricultural guidance. If symptoms spread, take clear photos and consult a KVK/agriculture officer before spraying."),
        "hi": ("मॉडल को इस तस्वीर में कोई परिचित रोग पैटर्न नहीं मिला। इसका अर्थ यह नहीं कि पौधा निश्चित रूप से रोग-मुक्त है; लक्षण शुरुआती या अस्पष्ट हो सकते हैं अथवा मॉडल की श्रेणियों में शामिल न हों।", "नई वृद्धि पर निगरानी रखें, संतुलित सिंचाई और पोषण दें तथा खेत की स्वच्छता बनाए रखें। लक्षण बढ़ें तो साफ तस्वीरें लेकर छिड़काव से पहले KVK/कृषि अधिकारी से सलाह लें।"),
        "mr": ("या प्रतिमेत मॉडेलला ओळखीचा रोग-पॅटर्न आढळला नाही. याचा अर्थ पीक निश्चितपणे रोगमुक्त आहे असा नाही; लक्षणे सुरुवातीची, अस्पष्ट किंवा मॉडेलच्या वर्गांबाहेरील असू शकतात.", "नवीन वाढीवर लक्ष ठेवा, संतुलित पाणी व पोषण द्या आणि शेत स्वच्छ ठेवा. लक्षणे वाढल्यास स्पष्ट छायाचित्रे घेऊन फवारणीपूर्वी KVK/कृषी अधिकाऱ्यांचा सल्ला घ्या.")
    }
}

# Disease-specific offline entries for every remaining class in the expanded model.
# English is the canonical fallback; Gemini provides a localized entry when reachable.
EXTRA_GUIDANCE = {
    "apple scab": ("Apple scab is a fungal disease caused by Venturia inaequalis. Olive-green to dark, velvety spots can develop on leaves and fruit; fruit lesions may become corky, cracked, or misshapen, and severe leaf infection can cause early leaf drop.", "Choose locally recommended resistant varieties where available, remove fallen infected leaves and mummified fruit, prune for good airflow, and monitor from early leaf growth. Confirm the diagnosis with a local horticulture/KVK specialist before any treatment."),
    "cedar apple rust": ("Cedar apple rust is a fungal disease that alternates between apple and certain juniper/cedar hosts. Apple leaves may show bright yellow-orange spots that later develop dark dots or tube-like structures, and fruit can also be affected.", "Use locally recommended resistant cultivars where available, monitor leaves after wet spring weather, and manage nearby alternate hosts only where practical and legally appropriate. Ask a local horticulture expert to confirm the disease before treatment."),
    "black rot": ("Black rot is a fungal disease whose symptoms depend on the host crop. On apple, leaves can develop brown 'frog-eye' spots, branches may form cankers, and fruit can rot and shrivel into dark mummies. On grape, leaf spots and mummified berries are common. The exact crop must be considered when interpreting symptoms.", "Remove mummified fruit and infected pruning material, prune cankered wood during suitable dry conditions, improve canopy airflow, and clean up crop debris. Use crop-specific local guidance to confirm the diagnosis and any management measures."),
    "powdery mildew": ("Powdery mildew is a fungal disease that often appears as white, flour-like patches on leaf surfaces and sometimes stems. Leaves may curl, yellow, or become distorted; severity and the causal fungus vary by crop.", "Use crop-appropriate resistant varieties where available, provide suitable plant spacing and airflow, avoid excessive nitrogen, remove heavily affected debris, and monitor early symptoms. Confirm the crop-specific disease before applying any treatment."),
    "cercospora leaf spot gray leaf spot": ("Maize gray leaf spot is caused by Cercospora fungi. It typically produces long, narrow, rectangular tan-to-grey lesions running parallel to leaf veins; lesions may merge and reduce the green leaf area, especially in humid, dense canopies.", "Use locally recommended resistant hybrids, rotate with non-host crops, manage infected maize residue where practical, and avoid dense, poorly ventilated stands. Confirm the diagnosis locally before considering disease-control products."),
    "esca (black measles)": ("Esca is a complex grapevine trunk disease. Leaves may show interveinal yellowing or reddish areas with green veins, sometimes in a tiger-stripe pattern; berries may develop dark speckling, and affected vines can decline or suddenly wilt. Symptoms vary by season and vine.", "Use healthy planting material, avoid creating large unnecessary pruning wounds, prune in dry weather where possible, remove severely affected wood safely, and seek advice from a local viticulture specialist. Do not rely on leaf appearance alone for confirmation."),
    "leaf blight (isariopsis leaf spot)": ("Grape Isariopsis leaf spot is a fungal leaf disease that can cause angular reddish-brown to dark lesions, often limited by leaf veins. Numerous lesions may merge, causing leaf tissue to dry and reducing functional leaf area.", "Maintain canopy airflow, remove heavily infected fallen leaves where practical, avoid prolonged leaf wetness, and monitor vines after humid weather. Confirm the cause with a local viticulture expert because several grape leaf disorders look similar."),
    "haunglongbing (citrus greening)": ("Citrus greening, also called Huanglongbing (HLB), is a serious bacterial disease associated with psyllid vectors. Typical signs include irregular blotchy yellow mottling that is not symmetrical across the leaf veins, yellow shoots, sparse growth, and small, lopsided fruit with uneven colour. Visual signs alone are not definitive.", "Use certified disease-free nursery plants, monitor and manage Asian citrus psyllid through locally recommended integrated methods, remove confirmed infected trees where advised, and avoid moving uncertified citrus material. Contact a local horticulture department/KVK promptly for confirmation; there is no simple curative spray."),
    "leaf scorch": ("Strawberry leaf scorch commonly causes many small purple-red to dark spots on leaves; centers may become tan or grey, and spots can merge so leaf edges look scorched. Other leaf spots and environmental stress can look similar.", "Use healthy planting stock, remove heavily infected old leaves and crop debris, maintain suitable spacing and airflow, avoid overhead irrigation when possible, and rotate or renew beds according to local recommendations. Confirm the disease before treatment."),
    "spider mites two-spotted spider mite": ("Two-spotted spider mites are tiny sap-feeding pests, not a fungal disease. Feeding causes fine pale stippling or bronzing on leaves; fine webbing may appear in heavier infestations, and leaves can dry under hot, dusty conditions.", "Inspect the undersides of leaves with a hand lens, conserve beneficial predatory mites, reduce dust and plant stress, and use locally recommended integrated pest management if populations rise. Confirm mites before selecting any pesticide; broad-spectrum sprays can worsen outbreaks by killing natural enemies."),
    "target spot": ("Tomato target spot, commonly associated with Corynespora cassiicola, produces brown circular to irregular leaf lesions that may develop concentric rings and yellow margins. Spots can occur on stems and fruit, and severe leaf infection can cause defoliation; symptoms overlap with other leaf diseases.", "Use healthy transplants, improve airflow and spacing, avoid prolonged leaf wetness, remove infected plant debris, and rotate crops where feasible. Have the disease confirmed locally before treatment."),
    "tomato mosaic virus": ("Tomato mosaic virus can cause mottled light- and dark-green mosaic patterns, leaf distortion, reduced leaf size, and stunted growth; fruit may ripen unevenly or show blemishes. Symptoms vary with cultivar and conditions and can resemble other viral or herbicide-injury problems.", "Use certified virus-free seed/transplants, wash hands and sanitize tools between plants, remove confirmed infected plants carefully, control weeds that can host the virus, and avoid tobacco contamination during handling. There is no curative fungicide for a virus; seek local diagnostic confirmation."),
    "alternaria leaf spot": ("Cotton Alternaria leaf spot is a fungal disease that may cause round brown lesions, often with concentric rings and yellow margins. Spots can enlarge or merge and lead to premature leaf drop when conditions favour disease.", "Use locally recommended varieties, maintain balanced crop nutrition, avoid prolonged leaf wetness where irrigation is controllable, manage infected crop debris, and monitor lower and older leaves. Confirm the cause locally before any chemical treatment."),
    "bacterial blight": ("Cotton bacterial blight is caused by Xanthomonas bacteria. Symptoms may include angular water-soaked leaf spots limited by veins, darkened veins or petioles, stem lesions sometimes called black arm, and infection of bolls. Symptoms can be confused with other leaf damage.", "Use certified healthy seed and locally recommended resistant varieties, follow seed-health advice, avoid moving infected residues between fields, and manage crop debris as recommended locally. Confirm the disease with a KVK/agriculture officer before treatment."),
    "fusarium wilt": ("Cotton Fusarium wilt is a soil-associated fungal disease that interferes with water transport. Plants may become stunted, yellow, and wilt, sometimes on one side first; cutting the stem may reveal brown vascular tissue. Drought, root injury, and other wilts can look similar.", "Use locally recommended tolerant varieties and healthy seed, maintain soil health and balanced irrigation, reduce root injury, and follow crop-rotation and soil-management advice suited to the local pathogen pressure. Confirm the cause before making field-wide decisions."),
    "verticillium wilt": ("Cotton Verticillium wilt is a soil-borne fungal disease that affects water movement. Lower leaves may develop interveinal yellowing, followed by wilt and necrosis; vascular tissue inside the stem can show brown discoloration. Symptoms are affected by temperature and can resemble Fusarium wilt or water stress.", "Use locally recommended tolerant varieties, clean equipment between affected fields, manage crop residues and rotation according to local guidance, and avoid moving infested soil. Seek local diagnostic confirmation because management differs among wilt causes."),

    "bacterial leaf blight": ("Soybean bacterial leaf blight can cause water-soaked to brown lesions, sometimes angular and limited by veins, and may lead to leaf yellowing or blighting. Other bacterial and fungal leaf spots can look similar, so a photo-based label is not a laboratory confirmation.", "Use clean seed and healthy planting material, avoid working in wet foliage, reduce splash from irrigation or rain where practical, rotate crops, and manage infected residues. Seek local KVK/agricultural university confirmation before making field-wide treatment decisions."),
    "dry leaf": ("Dry leaf is a broad visual condition rather than a single confirmed soybean disease. Leaves can dry from ageing, drought, nutrient imbalance, root stress, pests, or infection; the image class alone cannot establish the cause.", "Check soil moisture, drainage, recent weather, root condition, and whether symptoms affect old or new leaves. Maintain balanced irrigation and nutrition, inspect for pests and spreading lesions, and ask a local KVK to confirm the cause before applying any treatment."),
    "septoria brown spot": ("Soybean Septoria brown spot is a fungal leaf disease that commonly begins on lower leaves as small dark-brown spots, sometimes with yellowing around them. Symptoms may spread upward in wet weather and can resemble other leaf spots or nutrient stress.", "Use healthy seed and locally recommended varieties, rotate with non-host crops, manage infected crop residue as advised locally, avoid excessive canopy wetness, and monitor lower leaves after rainy periods. Confirm the diagnosis locally before treatment."),
    "vein necrosis": ("Vein necrosis describes damaged or dead tissue along leaf veins and is a symptom category, not by itself a definitive pathogen diagnosis. In soybean it can be associated with several disease or stress causes, so appearance alone cannot determine the underlying agent.", "Inspect the whole plant and nearby plants, note whether symptoms are spreading, check moisture and recent chemical exposure, and keep clear photographs. Avoid assuming a specific pathogen from this label; consult a local KVK or plant diagnostic laboratory before treatment."),
    "brown rust": ("Wheat brown rust is a fungal disease that produces small orange-brown to dark-brown powdery pustules, often scattered across leaf surfaces. Heavy infection can reduce green leaf area and grain filling; symptoms should be distinguished from yellow rust and other leaf disorders.", "Use locally recommended resistant varieties, monitor crops from early growth through grain filling, avoid excess nitrogen, and follow regional crop rotation and residue-management guidance. Confirm the rust type with a local agricultural expert before any disease-control treatment."),
    "yellow rust": ("Wheat yellow rust, also called stripe rust, is a fungal disease that typically forms yellow-orange pustules in narrow stripes along the leaf veins. It can spread quickly in cool, moist weather and may reduce photosynthesis and grain yield.", "Plant locally recommended resistant varieties, monitor fields during cool humid periods, avoid excessive nitrogen, and follow regional extension advice on surveillance and management. Seek prompt KVK confirmation if stripe-like yellow pustules appear."),
    "septoria": ("Wheat Septoria leaf blotch is a fungal disease that can cause elongated brown lesions with pale or grey centers and tiny dark fruiting bodies. Lesions often start on lower leaves and can move upward during wet weather, reducing green leaf area.", "Use healthy seed and locally recommended varieties, rotate crops where practical, manage infected cereal residue according to local guidance, and monitor lower leaves after prolonged wet weather. Confirm the disease locally before treatment."),
    "cercospora leaf spot": ("Tur (pigeonpea) Cercospora leaf spot is a fungal disease that can cause small round to irregular brown spots on leaves, sometimes with yellow margins. Spots may enlarge or merge in favourable humid conditions and can reduce healthy leaf area.", "Use healthy seed and locally recommended varieties, remove or manage infected residues as advised locally, rotate crops where practical, and monitor leaves after humid weather. Ask a local KVK or pulse-crop specialist to confirm the diagnosis before treatment."),
    "sterility mosaic disease": ("Pigeonpea sterility mosaic disease is a viral disease associated with a mite vector. Symptoms can include pale mosaic mottling, small leaves, excessive branching or bushy growth, stunting, and few or no flowers or pods. Early symptoms can be confused with other stresses.", "Use locally recommended tolerant varieties and healthy planting material, monitor for the mite vector, manage volunteer pigeonpea and weeds as advised locally, and seek prompt KVK guidance if mosaic symptoms appear. There is no simple curative spray for a viral disease."),
}

def _post_json(url, payload, key, timeout=15):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json", "x-goog-api-key": key}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))

def _clean_json(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start: raise ValueError("AI did not return a JSON description")
    return json.loads(text[start:end + 1])

def _offline_guidance(crop, disease, lang, message):
    lang = lang if lang in LANG else "en"
    crop_key = crop.strip().lower()
    key = disease.strip().lower()

    # These disease names are shared by crops but require different crop-specific guidance.
    if key == "early blight" and crop_key.startswith("potato"):
        potato_early = {
            "en": ("Potato early blight is commonly caused by Alternaria fungi. It often starts on older leaves as brown, target-like spots with concentric rings and yellowing. Severe infection can cause leaves to die early and reduce tuber yield; other leaf problems can look similar.", "Use healthy seed tubers and locally recommended varieties, rotate away from potato and tomato, remove volunteer plants and infected debris, maintain balanced nutrition, and avoid prolonged leaf wetness. Monitor lower leaves and confirm the diagnosis with a local KVK before treatment."),
            "hi": ("आलू का अर्ली ब्लाइट आमतौर पर अल्टरनेरिया फफूंद से जुड़ा होता है। पुराने पत्तों पर गोल भूरे धब्बे और उनमें लक्ष्य जैसे छल्ले दिख सकते हैं; आसपास पीलापन भी हो सकता है। गंभीर संक्रमण में पत्तियाँ समय से पहले सूख सकती हैं और कंद की उपज घट सकती है।", "स्वस्थ बीज-कंद और स्थानीय अनुशंसित किस्में अपनाएँ, आलू व टमाटर से अलग फसल चक्र रखें, स्वयं उगे पौधे और संक्रमित अवशेष हटाएँ, संतुलित पोषण दें और पत्तियों को लंबे समय तक गीला न रहने दें। उपचार से पहले KVK से पुष्टि कराएँ।"),
            "mr": ("बटाट्यावरील अर्ली ब्लाइट हा बहुधा अल्टरनेरिया बुरशीशी संबंधित असतो. जुन्या पानांवर एकमेकांत वर्तुळे असलेले तपकिरी डाग आणि भोवती पिवळेपणा दिसू शकतो. तीव्र प्रादुर्भावात पाने अकाली वाळून कंद उत्पादन कमी होऊ शकते.", "निरोगी बियाणे-कंद व स्थानिक शिफारस केलेले वाण वापरा, बटाटा व टोमॅटोपासून पीक फेरपालट करा, आपोआप उगवलेली झाडे व रोगट अवशेष काढा, संतुलित पोषण द्या आणि पाने दीर्घकाळ ओली राहू देऊ नका. उपचारापूर्वी KVK कडून खात्री करून घ्या.")
        }
        description, prevention = potato_early[lang]
    elif key == "bacterial spot" and crop_key.startswith("peach"):
        description = "Peach bacterial spot is caused by the bacterium Xanthomonas arboricola pv. pruni. Leaves may develop small angular purple-to-black spots that can drop out and leave a shot-hole appearance; fruit may show sunken dark spots and twigs can develop lesions."
        prevention = "Use healthy nursery stock and locally recommended less-susceptible varieties, prune and remove infected twigs where advised, avoid moving infected plant material, and maintain orchard sanitation. Ask a local horticulture expert to confirm the cause before treatment."
    elif key == "bacterial spot" and crop_key.startswith("bell pepper"):
        description = "Bacterial spot of pepper is caused by Xanthomonas bacteria. Leaves may develop small water-soaked spots that become brown and sometimes have yellow margins; fruit can develop raised, scabby lesions. Symptoms can resemble fungal leaf spots."
        prevention = "Use clean seed and healthy transplants, avoid handling plants while wet, reduce overhead irrigation, sanitize tools, remove infected debris, and rotate away from pepper and tomato. Confirm the diagnosis locally before treatment."
    elif key == "powdery mildew" and crop_key.startswith("wheat"):
        wheat_powdery = {
            "en": ("Wheat powdery mildew is a fungal disease that often appears as white, powdery patches on leaves and stems. Infected tissue may yellow or dry, and severe infection can reduce photosynthesis and grain filling. White surface growth alone is not a definitive diagnosis.", "Use locally recommended resistant varieties, avoid excessive nitrogen and overly dense stands, monitor lower leaves early, and follow regional extension advice. Confirm the disease with a local KVK before any chemical treatment.") ,
            "hi": ("गेहूँ का पाउडरी मिल्ड्यू एक फफूंदजनित रोग है, जिसमें पत्तियों और तनों पर सफेद चूर्ण जैसा जमाव दिख सकता है। प्रभावित ऊतक पीले पड़ सकते हैं या सूख सकते हैं और गंभीर संक्रमण दाना भरने को प्रभावित कर सकता है। केवल सफेद परत से रोग की पक्की पुष्टि नहीं होती।", "स्थानीय अनुशंसित प्रतिरोधी किस्में अपनाएँ, नाइट्रोजन की अधिकता और बहुत घनी फसल से बचें, निचली पत्तियों की जल्दी जाँच करें और क्षेत्रीय कृषि सलाह का पालन करें। रासायनिक उपचार से पहले KVK से पुष्टि कराएँ।"),
            "mr": ("गव्हावरील पावडरी मिल्ड्यू हा बुरशीजन्य रोग आहे. पानांवर आणि खोडांवर पांढऱ्या पिठासारखा थर दिसू शकतो. बाधित भाग पिवळे पडू शकतात किंवा वाळू शकतात; तीव्र प्रादुर्भावामुळे दाणे भरण्यावर परिणाम होऊ शकतो. केवळ पांढरा थर दिसल्याने रोगाची खात्री होत नाही.", "स्थानिक शिफारस केलेले रोगप्रतिकारक वाण वापरा, नायट्रोजनचा अतिरेक आणि अतिघन पीक टाळा, खालच्या पानांची लवकर तपासणी करा आणि स्थानिक कृषी सल्ला पाळा. रासायनिक उपचारापूर्वी KVK कडून खात्री करून घ्या.")
        }
        description, prevention = wheat_powdery[lang]
    elif key == "late blight" and crop_key.startswith("tomato"):
        tomato_late = {
            "en": ("Tomato late blight is caused by Phytophthora infestans, an oomycete that can spread rapidly in cool, wet conditions. Leaves may develop irregular water-soaked, dark green to brown lesions; pale growth may appear at lesion edges in humid weather. Stems and fruit can also develop dark, firm lesions.", "Monitor plants frequently during wet weather, use healthy transplants and locally recommended varieties, improve airflow, avoid prolonged leaf wetness, and remove infected material carefully. Because it can spread quickly, seek prompt KVK advice and use only locally registered products according to label directions if recommended."),
            "hi": ("टमाटर का लेट ब्लाइट Phytophthora infestans नामक जीव के कारण होता है और ठंडे, नम मौसम में तेजी से फैल सकता है। पत्तियों पर अनियमित पानी सोखे हुए जैसे गहरे हरे से भूरे धब्बे बन सकते हैं; नमी में किनारों पर हल्की सफेद वृद्धि दिख सकती है। तने और फलों पर भी गहरे धब्बे हो सकते हैं।", "नम मौसम में पौधों की बार-बार जाँच करें, स्वस्थ पौध लगाएँ, स्थानीय अनुशंसित किस्में अपनाएँ, हवा का आवागमन बढ़ाएँ और पत्तियों को लंबे समय तक गीला न रहने दें। यह तेजी से फैल सकता है, इसलिए शीघ्र KVK से सलाह लें और जरूरत होने पर केवल स्थानीय रूप से पंजीकृत उत्पाद लेबल के अनुसार इस्तेमाल करें।"),
            "mr": ("टोमॅटोवरील लेट ब्लाइट हा Phytophthora infestans मुळे होतो आणि थंड, दमट हवेत वेगाने पसरू शकतो. पानांवर अनियमित, पाणचट दिसणारे गडद हिरवे ते तपकिरी डाग पडू शकतात; दमट हवेत डागांच्या कडेला फिकट वाढ दिसू शकते. खोडे व फळांवरही गडद डाग पडू शकतात.", "दमट हवेत झाडांची वारंवार पाहणी करा, निरोगी रोपे व स्थानिक शिफारस केलेले वाण वापरा, हवा खेळती ठेवा आणि पाने दीर्घकाळ ओली राहू देऊ नका. रोग वेगाने पसरू शकतो; त्वरित KVK चा सल्ला घ्या आणि गरज असल्यास फक्त स्थानिक नोंदणीकृत उत्पादन लेबलनुसार वापरा.")
        }
        description, prevention = tomato_late[lang]
    else:
        all_guidance = {**GUIDANCE, **EXTRA_GUIDANCE}
        if key not in all_guidance:
            matched = next((name for name in all_guidance if name in key or key in name), None)
            if matched:
                key = matched
            else:
                # Never mislabel an unrecognised disease class with the healthy-class text.
                return {"status": "unavailable", "message": message, "description": "", "prevention": "", "sources": []}
        entry = all_guidance[key]
        if isinstance(entry, dict):
            description, prevention = entry.get(lang, entry.get("en", ("", "")))
        else:
            description, prevention = entry
    return {"status": "fallback", "message": message, "description": description, "prevention": prevention, "sources": []}

def offline_guidance(crop, disease, lang="en"):
    """Return disease-specific local guidance without making an external AI request."""
    return _offline_guidance(crop, disease, lang, "")

def research_disease(crop, disease, lang):
    lang = lang if lang in LANG else "en"
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return _offline_guidance(crop, disease, lang, "")
    model = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite").strip() or "gemini-3.1-flash-lite"
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    prompt = f"""You are an agricultural information assistant for farmers in Maharashtra, India. Write entirely in {LANG[lang]}.
The image classifier (not you) has predicted crop: {crop}; class: {disease}. Treat this as an unconfirmed machine-learning prediction. Do NOT re-analyse the image, change the predicted crop/disease, offer alternative diagnoses, or score confidence. Your only job is to explain the predicted disease and how to prevent/manage it safely.
Use Google Search grounding and prioritise ICAR, Maharashtra agricultural universities, KVKs, and Indian government agricultural sources. Return a single JSON object with exactly two string fields: description and prevention. description should explain the disease/cause and characteristic symptoms in useful detail. prevention should give practical integrated prevention steps. Do not prescribe pesticide products, doses, or chemical treatment. If chemical control might be needed, advise confirming current crop-specific registration and label directions with a local KVK/agriculture officer. If the predicted class is Healthy, explain that no familiar disease pattern was predicted (not proof of a disease-free plant) and give general preventive crop-care steps. Do not invent facts or sources."""
    try:
        response = _post_json(endpoint, {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "tools": [{"google_search": {}}], "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}}, key)
        candidate = (response.get("candidates") or [{}])[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        report = _clean_json("\n".join(p.get("text", "") for p in parts if p.get("text")))
        description, prevention = str(report.get("description", "")).strip(), str(report.get("prevention", "")).strip()
        if not description or not prevention:
            raise ValueError("AI response missing description/prevention")
        sources, seen = [], set()
        for chunk in (candidate.get("groundingMetadata") or {}).get("groundingChunks", []):
            web = chunk.get("web") or {}
            url = web.get("uri", "")
            title = web.get("title", "")
            if url.startswith("https://") and url not in seen:
                seen.add(url)
                sources.append({"title": (title or url)[:180], "url": url})
        return {"status": "ok", "message": "", "description": description[:5000], "prevention": prevention[:5000], "sources": sources[:6]}
    except urllib.error.HTTPError:
        return _offline_guidance(crop, disease, lang, "")
    except Exception:
        return _offline_guidance(crop, disease, lang, "")

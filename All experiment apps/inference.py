from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Path to your saved model
MODEL_DIR = "./biobart-mri"

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

# Example input (MRI findings)
test_text = test_texts = [
    """Mild bone marrow edema of the patella noted. Subchondral cystic change of the lateral
tibial spine noted with surrounding focal bone marrow edema. Mucoid degeneration of the anterior
cruciate ligament noted, otherwise grossly intact. Posterior cruciate ligament is grossly intact. Signal
attenuation of the medial collateral ligament noted. Major components of the lateral collateral
ligament complex are grossly contiguous. Popliteus tendon is grossly unremarkable. Mild myxoid
degeneration posterior horn lateral meniscus noted. Mild fraying of the peripheral posterior horn
medial meniscus noted. No acute meniscal tear. Mild patellar and quadriceps tendinosis noted.
Grade III chrondromalacia patella noted in the medial facet. Incomplete medial plica noted. No
pathologic suprapatellar effusion noted. Minimal deep infrapatellar bursal fluid noted. Minimal
amount of fluid noted in the lateral patellar synovial recess. Minimal subcutaneous medial preinfrapatellar bursitis noted. Semimembranosus tendinosis noted. Minimal amount of fluid noted in the
semimembranosus- gastrocnemius bursa.""",
    """No prior exams. Normal marrow stores are seen in the visualized osseous elements. The
vertebral bodies are normal in height and alignment and normal cervical lordosis is seen. The
craniovertebral junction is unremarkable.
Disc desiccation is noted at all levels visualized in the cervical and upper thoracic spine. There is
loss of disc height at C5-C6 and to a lesser degree at C6-C7. A small rounded focal lesion is seen
in the leftward inferior end plate of T3 which measures 7 mm with signal intensity cost with
fat.
C2-C3: No disc herniation or bulging. No canal or foraminal stenosis.
C3-C4: No disc herniation or bulging. No canal or foraminal stenosis.
C4-C5: No disc herniation or bulging. No canal or foraminal stenosis. Mild right facet
hypertrophy is noted.
C5-C6: A broad based mixed protrusion is seen which extends into the foramina and is slightly
prominent left paracentrally. It results in ventral thecal sac effacement and there is slight
flattening suggested of the leftward ventral aspect of the cord. In conjunction with
uncovertebral joint spurring and facet hypertrophy there is moderate to severe bilateral
foraminal stenosis. Mild central canal stenosis is noted.
C6-C7: Mild posterior disc bulging with uncovertebral joint spurring and facet hypertrophy
resulting in mild bilateral foraminal stenosis.
C7-T1: No disc herniation or bulging. No canal or foraminal stenosis.
Apart from the slight mass effect upon the leftward cord at C5-C6, the spinal cord is normal in
course, caliber, and signal intensity.
""",
    """Postoperative findings of posterior intrapedicular spinal fusion at L2-L3 noted. The L2-L3 disk is
preserved. Enhancing peridural fibrosis noted at L2-L3 level mildly deforming the thecal sac with
dominant extrinsic impression on the right lateral thecal sac. Non enhancing cystic foci noted
along the posterior elements representing small pseudomeningoceles. Postoperative fusion and
laminectomy noted at L4-L5 level with osseous fusion anteriorly. Osseous hypertrophy of the
posterior elements noted at L4 and L5. Lumbar lordosis is decreased. Multilevel endplate, disk and
facet degenerative changes noted. Conus medullaris terminates at approximately mid L1 vertebral
body level.
L1-L2 shows moderate broad-based disc bulging contributing to mild to moderate left greater than
right neuroforamina narrowing. Spinal canal is grossly patent. Approximately 2 mm L1 on L2
retrolisthesis noted.
L2-L3 shows moderate nonenhancing bi foraminal broad-based disk bulging contributing to mild-tomoderate right greater than left neural foramina narrowing. Moderate acquired spinal canal stenosis
noted due to enhancing peridural fibrosis with asymmetric more focal extrinsic impression on the
right lateral ventral thecal sac. Negligible spondylolisthesis of L2 on L3 noted.
L3-L4 level shows mild disk desiccation and height loss. Extraforaminal focal annular tears noted on
both sides. Spinal canal and foramina are patent.
L4-L5 level shows postoperative findings with partial fusion anteriorly with linear hyper intense signal
in the remaining intervertebral disk space. Spinal canal and foramina are patent. No gross thecal sac
deformity noted. Bilateral laminectomies noted.
L5-S1 level shows subtle left central broad-based disk protrusion. Spinal canal and foramina are
patent. No gross thecal sac deformity. Bilateral laminectomies noted.
Ferromagnetic susceptibility artifact noted along the mid posterior back spanning from L2 through
S2.
No suspicious prevertebral or posterior paraspinal soft tissue signal abnormality noted.
Mild subchondral sclerosis of the included sacroiliac joints noted.
Incidental note of overdistended bladder.
""",
    """Vertebral body signal and vertebral body height are preserved. There is a somewhat steep lumbar
lordosis. The overall canal size is unremarkable. There is mild-to-moderate multilevel lower thoracic
disk degeneration at the periphery of the field of view of questionable significance.
At L1-L2, there is some desiccation with mild bi-foraminal disk bulging, right greater than left, but
without distinct focal disk herniation.
There is desiccation L2-L3 without evidence of disk herniation.
At L3-L4, there is moderate disk degeneration with some loss of disk signal and disk height. There is
mild bi-foraminal disk bulging, right greater than left without focal disk herniation.
At L4-L5, there is desiccation and some facet joint DJD, but there is no evidence of disk herniation
and neurofamina compromise.
At L5-S1, there is disk degeneration with loss of disk signal and disk height. There is a mild
spondylolisthesis of L5 upon S1. This appears to be due to facet joint DJD/laxity. There is mildtomoderate broad-based disk protrusion with shallow midline/left paramedian disk herniation. Left
neural foramina is somewhat narrowed when compared to the right.
Sagittal images obtained in flexion reveal limited range of motion. There was no evidence of
instability, pathologic offset or alteration in posterior disk margin.
Sagittal images obtained in extension, also reveal limited range of motion without evidence of
instability, pathologic offset or alteration in posterior disk margin.
""",
    """Vertebral body signal and vertebral body height are preserved. The overall canal size is
unremarkable. The conus medullaris is clear. At L1-L2, L2-L3, L3-L4 and L4-L5 there is normal disk
signal and disk height. A few images suggest mild disk bulging to the left at L4-L5.
At L5-S1, there is disk degeneration with loss of disk signal and disk height. There is mild to
moderate broad-based disk protrusion impinging upon the anterior aspect of the dural sac and
narrowing both neuroforamina and clinical correlation is advised regarding the status of both L5
nerve roots.
Sagittal images obtained in flexion reveal normal range of motion without evidence of instability or
pathologic offset. Disk protrusion is somewhat more apparent than in the neutral/sitting position.
Sagittal images obtained in extension reveal normal range of motion without evidence of instability or
pathologic offset. Disk protrusion is also somewhat more apparent than in the neutral/sitting position.""",
    """Skin marker is noted on the left side of the neck at the level of the C4/5 intervertebral disc space.
Cervical spine alignment is normal. Vertebral bodies are preserved in signal and height.
Intervertebral disc spaces are normal in height with mild loss of intervertebral disc space signal at
C2/3 through C5/6, consistent with desiccation. There are shallow disc-osteophyte complex is at
C4/5 and C5/6. The facet joints are normal. There is no evidence of spinal canal or foraminal
stenosis. The visualized spinal cord and posterior fossa are normal in signal and contour. There is
no evidence of intra or extradural mass lesion.
There is a T2 and T1 hyperintense encapsulated mass in the left posterior cervical space, between
the sternocleidomastoid muscle and left paraspinal muscles. The mass is posterior to the carotid
space without appreciable mass effect in the carotid space. The mass measures approximately 4 x
1.3 x 6 cm in AP, lateral and craniocaudad dimensions respectively. The mass follows normal fat in
signal intensity. This most likely represents an incidental lipoma. Consider further evaluation with
contrast enhanced CT or MRI to exclude an enhancing component which may indicate neoplasia.
Remaining cervical soft tissues are normal.""",
    """Marker overlies the medial aspect of the wrist. No acute osseous fracture noted.
Degenerative arthrosis of the included wrist joints noted with subchondral cystic and/or erosive
change of the carpal bones. Chronic fracture of the hook of the hamate noted. Narrowing of the
lunotriquetral joint noted. No increased signal is noted in the lunotriquetral joint space. There is a
slight dorsal subluxation of the distal ulna in relation to the sigmoid notch of the distal radius and
apparent negligible distal radioulnar joint diastasis which is likely related to pronation of the hand.
There is slight medial subluxation of the extensor carpi ulnaris tendon likely due to pronation.
Underlying extensor carpi ulnaris tendinosis noted. No acute full thickness extensor or flexor tendon
tear noted. An approximately 7 x 7 x 15 mm dimension lobulated cystic structure is noted along the
radiopalmar aspect of the wrist associated with the long radiolunate ligament. An approximately 4 x 8
mm dimension lobulated T2 hyperintense focus is identified impressing the deep surface of the
carpal tunnel along the ulnar aspect of the wrist associated with the radioscaphoidcapitate ligament.
An approximately 3 mm cystic structure is associated with the volar radioulnar ligament. Signal
attenuation of the ulnar aspect dorsal radioulnar ligament noted. Focal signal attenuation/fraying of
the radial attachment of the triangular fibrocartilage complex noted. Pisotriquetral joint synovitis
noted. The median nerve is enlarged and shows increased STIR signal. No increased T2 signal of
the nerve noted. The median nerve measures 6 x 6 mm dimension at the level of the distal forearm
and 6 x 5 mm dimension within the carpal tunnel at the level of the hook of the hamate. Slight palmar
bowing of the flexor retinaculum noted. Mild fatty infiltration of the included pronator quadratus
muscle noted. Mild fatty replacement of the hypothenar and thenar musculature noted.
""",
    """There is no fracture or abnormal bone marrow edema. No suspicious bone lesion is noted.
There is positive ulnar variance. There is tear of the triangular fibrocartilage complex near radial
attachment. The, scapholunate ligament, the lunotriquetral ligament are intact.
The flexor and extensor tendons about the wrist are intact. Small joint effusion is noted in radiocarpal
joint with a 1.5 x 1.4 x 0.2 cm cystic structure palmar to the radiocarpal joint, likely representing
synovial cyst. There is a 3 mm cystic structure dorsal to the capitate scaphoid joint, which may
represent a small ganglion cyst or synovial cyst.
There are osteoarthritic changes in the first carpometacarpal joint with joint space narrowing, small
marginal osteophytes and small joint effusion. No evidence of soft tissue mass is noted.
The median and ulnar nerves are unremarkable. The carpal tunnel is intact.
There is a 1.5 x 0.7 x 0.2 cm lobulated cystic structure palmar to the second and third
carpometacarpal joints, which may represent a ganglion cyst or synovial cyst.""",
    #  Add more test cases here
]


# Tokenize
inputs = tokenizer(
    test_texts,
    return_tensors="pt",
    max_length=1024,
    truncation=True,
    padding=True   # 👈 ensures equal length
)

# Generate summary (impression)
summary_ids = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=160,
    min_length=40,
    num_beams=4,
    early_stopping=True
)

# Decode
for i, summary in enumerate(summary_ids):
    print(f"\n🔹 Test case {i+1} impression:\n", tokenizer.decode(summary, skip_special_tokens=True))
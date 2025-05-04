import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:novalabs/Details3.dart';
import 'package:shared_preferences/shared_preferences.dart';
class SpecificQuestions extends StatefulWidget {
  final Map<String, dynamic>? result;
  final String image;
  final String skin;
  const SpecificQuestions({super.key,required this.result,required this.image,required this.skin});

  @override
  State<SpecificQuestions> createState() => _SpecificQuestionsState();
}

List<String> q1=<String>["face","neck","chest","back","head","elbow","knees","scalp","lower back"];
List<List<String>> l1 = [["face","neck","chest","back"],["face","head","neck"],["elbows","knees","scalp","lower back"]];
List<String> list1 = <String>[
  'Common on the face, neck, chest, and back with no specific symmetry.',
  'Found on sun-exposed areas, such as the face, head, and neck.',
  'Symmetrically on elbows, knees, scalp, lower back, and other areas.'
];
List<List<String>>l2 = [['painful'],['painful'],['doesnot hurt']];
List<String> list2 = <String>[
  'painful',
  'doesnot hurt'
];
List<String> list3 = <String>[
  'No significant changes in skin texture, rare ulceration or bleeding',
  'Skin texture may be oily, with no significant changes',
  'Possible changes in texture, with itching and occasional pain'
];
List<String> q3 = <String>['No significant changes in skin texture','rare ulceration or bleeding','Skin texture may be oily','with no significant changes','Possible changes in texture','with itching and occasional pain'];
List<List<String>> l3 = [['No significant changes in skin texture','rare ulceration or bleeding'],['Skin texture may be oily','with no significant changes'],['Possible changes in texture','with itching and occasional pain']];
List<String> list4 = <String>[
  'whiteheads/blackheads/inflamed red',
  'shiny,skin colored bump,translucent',
  'dry,thick,patches'
];
List<String> q4 = <String>['whiteheads','blackheads','inflamed red','shiny','skin colored bump','translucent','dry','thick','patches'];
List<List<String>> l4 = [['whiteheads','blackheads','inflamed red'],['shiny','skin colored bump','translucent'],['dry','thick','patches']];
List<String> list5 = <String>[
  'Lesions may be tender, but severe pain is uncommon. Itching can occur',
  'Rare severe pain, but itching can occur, especially with inflammation',
  'Itching is common, and lesions can be painful, especially if joints are involved',
];
List<String> q5 = <String>['Lesions may be tender','but severe pain is uncommon. Itching can occur','Rare severe pain','itching can occur','especially with inflammation','Itching is common','lesions can be painful','especially if joints are involved'];
List<List<String>> l5 = [['Lesions may be tender','but severe pain is uncommon. Itching can occur'],['Rare severe pain','itching can occur','especially with inflammation'],['Itching is common','lesions can be painful','especially if joints are involved']];
List<String> list6 = <String>[
  'comedness',
  'pearl/waxy bump',
  'red/flaky patches'
];
List<String> q6 = <String>['comedness','pearl','waxy bump','red','flaky patches'];
List<List<String>> l6 = [['comedness'],['pearl','waxy bump'],['red','flaky patches']];


class _SpecificQuestionsState extends State<SpecificQuestions> {
  String? language;
  String dropdown1 = q1.first;
  String dropdown2= list2.first;
  String dropdown3 = q3.first;
  String dropdown4 = q4.first;
  String dropdown5 = q5.first;
  String dropdown6 = q6.first;
  Map<String, String> answers = {};

  void initState() {
    super.initState();
    getLanguage();

  }

  Future<String?> getLanguage() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    setState(() {
      language = prefs.getString("userLanguage");
    });
    return null;
  }
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SingleChildScrollView(
        child: Container(
          child: Column(
            children: [
              Container(
                height: MediaQuery.of(context).size.height * 0.3,
                width: MediaQuery.of(context).size.width,
                child: Padding(
                  padding:
                  const EdgeInsets.only(top: 100.0, left: 10, right: 10),
                  child: Column(
                    children: [
                      Text(
                        language == 'hindi'
                            ? "स्वास्थ्य मूल्यांकन फॉर्म"
                            : "Disease Identification Form",
                        style: GoogleFonts.poppins(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
                      Padding(
                        padding: const EdgeInsets.only(top: 12.0),
                        child: Text(
                          language == 'hindi'
                              ? "हमारे स्वास्थ्य मूल्यांकन फॉर्म के माध्यम से महत्वपूर्ण स्वास्थ्य विवरण साझा करें। आपका इनपुट आपके कुल स्वास्थ्य के समग्र समझन के लिए व्यक्तिगत देखभाल को मार्गदर्शित करता है।"
                              : "Share vital health details through our Health Assessment Form. Your input guides personalized care for a holistic understanding of your well-being.",
                          style: GoogleFonts.poppins(
                            fontSize: 12,
                            color: Colors.white,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                decoration: BoxDecoration(
                  color: Color.fromRGBO(86, 128, 227, 1),
                  borderRadius: BorderRadius.only(
                    bottomLeft: Radius.circular(20),
                    bottomRight: Radius.circular(20),
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.only(left: 20, top: 20, right: 20),
                child: Container(
                  child: Column(
                    children: [
                      Padding(
                        padding: const EdgeInsets.only(top: 10.0),
                        child: Row(
                          children: [
                            Container(
                              width: MediaQuery.of(context).size.width * 0.8955,
                              child: Text(
                                language == 'hindi'
                                    ? "वितरण और स्थिति:"
                                    : "Distribution and Location:",
                                style: GoogleFonts.poppins(fontSize: 15),
                              ),
                            ),
                          ],
                          mainAxisAlignment: MainAxisAlignment.start,
                        ),
                      ),
                      Column(
                        children: [
                          DropdownMenu<String>(
                            initialSelection:q1.first,
                            // language == 'hindi' ? h2.first : list1.first,
                            width: MediaQuery.of(context).size.width * 0.90,
                            onSelected: (String? value) {
                              // This is called when the user selects an item.
                              setState(() {
                                dropdown1 = value!;
                              });
                            },
                            dropdownMenuEntries:  q1
                                .map<DropdownMenuEntry<String>>((String value) {
                              return DropdownMenuEntry<String>(
                                  value: value, label: value);
                            }).toList(),
                          ),
                        ],
                      ),
                      Padding(
                        padding: const EdgeInsets.only(top: 10.0),
                        child: Row(
                          children: [
                            Container(
                              width: MediaQuery.of(context).size.width * 0.8955,
                              child: Text(
                                language == 'hindi'
                                    ? "दर्द का कारण है? "
                                    : "It causes pain?",
                                style: GoogleFonts.poppins(fontSize: 15),
                              ),
                            ),
                          ],
                          mainAxisAlignment: MainAxisAlignment.start,
                        ),
                      ),
                      Column(
                        children: [
                          DropdownMenu<String>(
                            initialSelection: list2.first,
                            width: MediaQuery.of(context).size.width * 0.90,
                            onSelected: (String? value) {
                              // This is called when the user selects an item.
                              setState(() {
                                dropdown2= value!;
                              });
                            },
                            dropdownMenuEntries: list2
                                .map<DropdownMenuEntry<String>>((String value) {
                              return DropdownMenuEntry<String>(
                                  value: value, label: value);
                            }).toList(),
                          ),
                        ],
                      ),

                      Padding(
                        padding: const EdgeInsets.only(top: 10.0),
                        child: Row(
                          children: [
                            Container(
                              width: MediaQuery.of(context).size.width * 0.8955,
                              child: Text(
                                language == 'hindi'
                                    ? "अन्य त्वचा परिवर्तन:"
                                    : "Other Skin Changes:",
                                style: GoogleFonts.poppins(fontSize: 15),
                              ),
                            ),
                          ],
                          mainAxisAlignment: MainAxisAlignment.start,
                        ),
                      ),
                      Column(
                        children: [
                          DropdownMenu<String>(
                            initialSelection: q3.first,
                            width: MediaQuery.of(context).size.width * 0.90,
                            onSelected: (String? value) {
                              // This is called when the user selects an item.
                              setState(() {
                                dropdown3= value!;
                              });
                            },
                            dropdownMenuEntries: q3
                                .map<DropdownMenuEntry<String>>((String value) {
                              return DropdownMenuEntry<String>(
                                  value: value, label: value);
                            }).toList(),
                          ),
                        ],
                      ),
                      Padding(
                        padding: const EdgeInsets.only(top: 10.0),
                        child: Row(
                          children: [
                            Container(
                              width: MediaQuery.of(context).size.width * 0.8955,
                              child: Text(
                                language == 'hindi'
                                    ? "उपस्थिति:"
                                    : "Appearance",
                                style: GoogleFonts.poppins(fontSize: 15),
                              ),
                            ),
                          ],
                          mainAxisAlignment: MainAxisAlignment.start,
                        ),
                      ),
                      Column(
                        children: [
                          DropdownMenu<String>(
                            initialSelection: q4.first,
                            width: MediaQuery.of(context).size.width * 0.90,
                            onSelected: (String? value) {
                              // This is called when the user selects an item.
                              setState(() {
                                dropdown4= value!;
                              });
                            },
                            dropdownMenuEntries: q4
                                .map<DropdownMenuEntry<String>>((String value) {
                              return DropdownMenuEntry<String>(
                                  value: value, label: value);
                            }).toList(),
                          ),
                        ],
                      ),
                      Padding(
                        padding: const EdgeInsets.only(top: 10.0),
                        child: Row(
                          children: [
                            Container(
                              width: MediaQuery.of(context).size.width * 0.8955,
                              child: Text(
                                language == 'hindi'
                                    ? "दर्द या खुजली का उपस्थिति:"
                                    : "Presence of Pain or Itching:",
                                style: GoogleFonts.poppins(fontSize: 15),
                              ),
                            ),
                          ],
                          mainAxisAlignment: MainAxisAlignment.start,
                        ),
                      ),
                      Column(
                        children: [
                          DropdownMenu<String>(
                            initialSelection: q5.first,
                            width: MediaQuery.of(context).size.width * 0.90,
                            onSelected: (String? value) {
                              // This is called when the user selects an item.
                              setState(() {
                                dropdown5 = value!;
                              });
                            },
                            dropdownMenuEntries:  q5
                                .map<DropdownMenuEntry<String>>((String value) {
                              return DropdownMenuEntry<String>(
                                  value: value, label: value);
                            }).toList(),
                          ),
                        ],
                      ),

                      Padding(
                        padding: const EdgeInsets.only(top: 10.0),
                        child: Row(
                          children: [
                            Container(
                              width: MediaQuery.of(context).size.width * 0.8955,
                              child: Text(
                                language == 'hindi'
                                    ? "विशेषता:"
                                    : "Distinguishing feature",
                                style: GoogleFonts.poppins(fontSize: 15),
                              ),
                            ),
                          ],
                          mainAxisAlignment: MainAxisAlignment.start,
                        ),
                      ),
                      Column(
                        children: [
                          DropdownMenu<String>(
                            initialSelection: q6.first,
                            width: MediaQuery.of(context).size.width * 0.90,
                            onSelected: (String? value) {
                              // This is called when the user selects an item.
                              setState(() {
                                dropdown6 = value!;
                              });
                            },
                            dropdownMenuEntries: q6
                                .map<DropdownMenuEntry<String>>((String value) {
                              return DropdownMenuEntry<String>(
                                  value: value, label: value);
                            }).toList(),
                          ),
                        ],
                      ),

                      Padding(
                        padding: const EdgeInsets.only(top: 20.0,bottom: 20),
                        child: Container(
                          width: MediaQuery.of(context).size.width*0.9,
                          decoration: BoxDecoration(
                              color:Color.fromRGBO(86, 128, 227, 1),
                              borderRadius: BorderRadius.circular(20)
                          ),
                          child: TextButton(
                            onPressed: ()async{
                              List<int> prob=[0,0,0];
                              int val = 0;
                              for(int i=0;i<3;i++){
                                if(l1[i].contains(dropdown1))
                                  prob[i]+=1;
                              }

                              for(int i=0;i<3;i++){
                                if(l2[i].contains(dropdown2))
                                  prob[i]+=1;
                              }
                              for(int i=0;i<3;i++){
                                if(l3[i].contains(dropdown3))
                                  prob[i]+=1;
                              }
                              for(int i=0;i<3;i++){
                                if(l4[i].contains(dropdown4))
                                  prob[i]+=1;
                              }
                              for(int i=0;i<3;i++){
                                if(l5[i].contains(dropdown5))
                                  prob[i]+=1;
                              }

                              for(int i=0;i<3;i++){
                                if(l6[i].contains(dropdown6))
                                  prob[i]+=1;
                              }

                              double prob1=(prob[0]/6)*0.4;
                              double prob2=(prob[1]/6)*0.4;
                              double prob3=(prob[2]/6)*0.4;
                              print(prob1);
                              print(prob2);
                              print(prob3);
                              widget.result!["result"][0][1]=(double.parse(widget.result!["result"][0][1])+prob1).toString();
                              widget.result!["result"][1][1]=(double.parse(widget.result!["result"][1][1])+prob2).toString();
                              widget.result!["result"][2][1]=(double.parse(widget.result!["result"][2][1])+prob3).toString();
                              String disease;

                              print(widget.result!["result"][0][1]);
                              print(widget.result!["result"][1][1]);
                              print(widget.result!["result"][2][1]);

                              if((double.parse(widget.result!["result"][0][1])>double.parse(widget.result!["result"][1][1]))&&(double.parse(widget.result!["result"][0][1])>double.parse(widget.result!["result"][2][1])))
                                disease = widget.result!["result"][0][0].toString();
                              else if(double.parse(widget.result!["result"][1][1])>double.parse(widget.result!["result"][2][1]))
                                disease = widget.result!["result"][1][0].toString();
                              else
                                disease =  widget.result!["result"][2][0].toString();


                              Navigator.of(context).push(MaterialPageRoute(builder: (context)=>Details3(result:widget.result,image:widget.image, skin: widget.skin,disease:disease)));
                            },
                            child: Text(language=='hindi'?"अगला कदम":"Next Step",style: GoogleFonts.poppins(color: Colors.white,fontSize: 16,fontWeight: FontWeight.bold),),
                          ),
                        ),
                      )
                      // Repeat similar changes for other text in your code
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
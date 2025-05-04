import 'dart:convert';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:novalabs/Detials.dart';
import 'package:novalabs/camera.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:http/http.dart' as http;

class Questions extends StatefulWidget {
  List<CameraDescription>? cameras;
  final String? user;
  final String skin;
  // final double temp;
  // final String from;
  Questions({super.key,required this.cameras,required this.user,required this.skin});

  @override
  State<Questions> createState() => _QuestionsState();
}
List<String> list1 = <String>[
  'Yes',
  'No'
];
List<String> list2 = <String>[
  'Yes',
  'No'
];
List<String> list6 = <String>[
  'Yes',
  'No'
];
List<String> list3 = <String>[
  'Thick',
  'Oil',
  'Dry'
];
List<String> list4 = <String>[
  'itching',
  'burning',
  'Pimples',
  'Blackhead',
  'Whitehead',
  'patches'
];
List<String> list5 = <String>[
  'Yes',
  'No'
];
List<String> list7= <String>[
  'Yes',
  'No'
];
List<String> list8= <String>[
  'Legs',
  'allparts',
  'face',
  'neck',
  'chest'
];
class _QuestionsState extends State<Questions> {
  String? language;
  String dropdown1 = list1.first;
  String dropdown2= list2.first;
  String dropdown3 = list3.first;
  String dropdown4 = list4.first;
  String dropdown5 = list5.first;
  String dropdown7= list7.first;
  String dropdown6 = list6.first;
  String dropdown8= list8.first;
  Map<String, String> answers = {};
  late File _image;

  final picker = ImagePicker();
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
                            : "Health Assessment Form",
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
                                    ? "क्या आपके पास कोई पूर्व मौजूदा चिकित्सा स्थिति है?"
                                    : "Do you have any pre-existing medical conditions?",
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
                            initialSelection:list1.first,
                            // language == 'hindi' ? h2.first : list1.first,
                            width: MediaQuery.of(context).size.width * 0.90,
                            onSelected: (String? value) {
                              // This is called when the user selects an item.
                              setState(() {
                                dropdown1 = value!;
                              });
                            },
                            dropdownMenuEntries:  list1
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
                                    ? "क्या आप वर्तमान में कोई दवाएं ले रहे हैं?"
                                    : "Are you currently taking any medications?",
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
                                    ? "क्या आपने पिछले में समान त्वचा समस्याएं अनुभव की हैं?"
                                    : "Have you experienced similar skin issues in the past?",
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
                            initialSelection: list6.first,
                            width: MediaQuery.of(context).size.width * 0.90,
                            onSelected: (String? value) {
                              // This is called when the user selects an item.
                              setState(() {
                                dropdown6= value!;
                              });
                            },
                            dropdownMenuEntries: list6
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
                                    ? "क्या त्वचा समस्या किसी दर्द, खुजली या असहिष्णुता के साथ जुड़ी है?"
                                    : "What is the nature of your skin? Are there any specific characteristics, conditions, or concerns that you would like to share or discuss related to your skin's health or appearance?",
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
                            initialSelection: list3.first,
                            width: MediaQuery.of(context).size.width * 0.90,
                            onSelected: (String? value) {
                              // This is called when the user selects an item.
                              setState(() {
                                dropdown3= value!;
                              });
                            },
                            dropdownMenuEntries: list3
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
                                    ? "क्या आपके आस-पास की परिस्थितियों या जीवनशैली में हाल ही में कोई परिवर्तन हुआ है?"
                                    : "Can you describe any symptoms or issues you're currently experiencing with your skin?",
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
                            initialSelection: list4.first,
                            width: MediaQuery.of(context).size.width * 0.90,
                            onSelected: (String? value) {
                              // This is called when the user selects an item.
                              setState(() {
                                dropdown4 = value!;
                              });
                            },
                            dropdownMenuEntries:  list4
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
                                    ? "क्या आपके परिवार में किसी को त्वचा रोग या स्थिति का इतिहास है?"
                                    : "Does anyone in your family have a history of skin diseases or conditions?",
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
                            initialSelection: list5.first,
                            width: MediaQuery.of(context).size.width * 0.90,
                            onSelected: (String? value) {
                              // This is called when the user selects an item.
                              setState(() {
                                dropdown5 = value!;
                              });
                            },
                            dropdownMenuEntries: list5
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
                                    ? "क्या आपको किसी भी एलर्जी का इलाज है, जिसमें आपको खाद्य या पर्यावरण से संबंधित एलर्जी शामिल है?"
                                    : "Have you had any recent changes in your environment or lifestyle?",
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
                            initialSelection:
                            list7.first,
                            width: MediaQuery.of(context).size.width * 0.90,
                            onSelected: (String? value) {
                              // This is called when the user selects an item.
                              setState(() {
                                dropdown7 = value!;
                              });
                            },
                            dropdownMenuEntries: list7
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
                                    ? "क्या आपको त्वचा समस्या के अलावा कोई अन्य लक्षण हैं?"
                                    : "Could you specify the location or affected region on your body where you are experiencing skin-related concerns or symptoms?",
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
                            initialSelection: list8.first,
                            width: MediaQuery.of(context).size.width * 0.90,
                            onSelected: (String? value) {
                              // This is called when the user selects an item.
                              setState(() {
                                dropdown1 = value!;
                              });
                            },
                            dropdownMenuEntries: list8
                                .map<DropdownMenuEntry<String>>((String value) {
                              return DropdownMenuEntry<String>(
                                  value: value, label: value);
                            }).toList(),
                          ),
                        ],
                      ),
                      // Padding(
                      //   padding: const EdgeInsets.only(top: 10.0),
                      //   child: Row(
                      //     children: [
                      //
                      //       Container(
                      //         width: MediaQuery.of(context).size.width * 0.8955,
                      //         child: Text(
                      //           language == 'hindi'
                      //               ? "क्या स्थिति किसी निश्चित स्थिति या दिन के किसी समय में बिगड़ती है?"
                      //               : "Does the condition worsen in certain situations or times of the day?",
                      //           style: GoogleFonts.poppins(fontSize: 15),
                      //         ),
                      //       ),
                      //     ],
                      //     mainAxisAlignment: MainAxisAlignment.start,
                      //   ),
                      // ),
                      // Column(
                      //   children: [
                      //     TextField(
                      //       onChanged: (String value) {
                      //         print('Entered text: $value');
                      //       },
                      //       controller: t9,
                      //       decoration: InputDecoration(
                      //         hintText: language == 'hindi'
                      //             ? 'यहां लिखें...'
                      //             : 'Type here...',
                      //         border: OutlineInputBorder(),
                      //       ),
                      //     ),
                      //   ],
                      // ),
                      //
                      // Padding(
                      //   padding: const EdgeInsets.only(top: 10.0),
                      //   child: Row(
                      //     children: [
                      //       Container(
                      //         width: MediaQuery.of(context).size.width * 0.8955,
                      //         child: Text(
                      //           language == 'hindi'
                      //               ? "क्या आपने त्वचा समस्या के लिए कोई बिना डॉक्टर के सुझाए गए उपचार या घरेलू उपाय किए हैं?"
                      //               : "Have you tried any over-the-counter treatments or home remedies for the skin problem?",
                      //           style: GoogleFonts.poppins(fontSize: 15),
                      //         ),
                      //       ),
                      //     ],
                      //     mainAxisAlignment: MainAxisAlignment.start,
                      //   ),
                      // ),
                      // Column(
                      //   children: [
                      //     TextField(
                      //       onChanged: (String value) {
                      //         print('Entered text: $value');
                      //       },
                      //       controller: t10,
                      //       decoration: InputDecoration(
                      //         hintText: language == 'hindi'
                      //             ? 'यहां लिखें...'
                      //             : 'Type here...',
                      //         border: OutlineInputBorder(),
                      //       ),
                      //     ),
                      //   ],
                      // ),
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

                              String type = "Type VI";

                              if(language=='hindi'){
                                answers['q1'] =
                                    dropdown1.toString();
                                answers['q2'] =
                                    dropdown2.toString();
                                answers['q3'] =
                                    dropdown6.toString();
                                answers['nature'] =
                                    dropdown3.toString();
                                answers['symptoms'] =
                                    dropdown4.toString();
                                answers['genetics'] =
                                    dropdown5.toString();
                                answers['environment'] =
                                    dropdown7.toString();
                                answers['location'] =
                                    dropdown8.toString();
                                answers['Skin Type'] = type.toString();

                                _showMyDialog2(widget.cameras, widget.user, context, widget.skin,answers);
                              }else {
                                answers['q1'] =
                                    dropdown1.toString();
                                answers['q2'] =
                                    dropdown2.toString();
                                answers['q3'] =
                                    dropdown6.toString();
                                answers['nature'] =
                                    dropdown3.toString();
                                answers['symptoms'] =
                                    dropdown4.toString();
                                answers['genetics'] =
                                    dropdown5.toString();
                                answers['environment'] =
                                    dropdown7.toString();
                                answers['location'] =
                                    dropdown8.toString();
                                answers['Skin Type'] = type.toString();

                                _showMyDialog2(widget.cameras, widget.user, context, widget.skin,answers);
                                print(answers);
                              }

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
  Future<void> _showMyDialog2(List<CameraDescription>? cameras, String? type,
      BuildContext context,String dropdown, Map<String, String> answers) async {
    bool _isloading = false;
    return showDialog<void>(
      context: context,
      barrierDismissible: false, // user must tap button!
      builder: (BuildContext context) {
        return AlertDialog(
          title: Center(child: const Text('Choose')),
          content:  Container(
              height: 158,
              child: Column(
                children: [

                  Column(
                    children: [
                      Center(
                        child: Container(
                          width: 152,
                          child: OutlinedButton(

                            child: Row(
                              children: [
                                Icon(Icons.browse_gallery),
                                Padding(
                                  padding: const EdgeInsets.all(8.0),
                                  child: Text(language=='hindi'?"गैलरी":"Gallery",style: GoogleFonts.roboto(fontSize: 20,color: Colors.black),),
                                )
                              ],
                              mainAxisAlignment: MainAxisAlignment.center,
                            ),

                            onPressed: ()async{

                              setState(() {
                                _isloading= true;
                              });
                              await getImageFromGallery(dropdown,answers);
                              setState(() {
                                _isloading= false;
                              });
                            },
                          ),

                        ),
                      ),
                      Padding(
                        padding: const EdgeInsets.only(top:10.0),
                        child: Center(
                          child: Container(
                            width: 158,
                            child: OutlinedButton(

                              child: Row(
                                children: [
                                  Icon(Icons.camera_alt),
                                  Padding(
                                    padding: const EdgeInsets.all(8.0),
                                    child: Text(language=='hindi'?"कैमरा":"Camera",style: GoogleFonts.roboto(fontSize: 20,color: Colors.black),),
                                  )
                                ],
                                mainAxisAlignment: MainAxisAlignment.center,
                              ),

                              onPressed: ()async{
                                Navigator.push(
                                    context,
                                    MaterialPageRoute(
                                        builder: (context) => TakePictureScreen(
                                            cameras: cameras,
                                            user: type,
                                            skin: dropdown,
                                            questions: answers,
                                        )));
                              },
                            ),

                          ),
                        ),
                      ),
                      if(_isloading==true)
                        Padding(
                          padding: const EdgeInsets.all(8.0),
                          child: CircularProgressIndicator(),
                        ),
                    ],
                  ),
                ],
              )
          ),

        );
      },
    );
  }
  Future getImageFromGallery(String dropdown,Map<String, String> answers) async {
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);

    setState(() {
      if (pickedFile != null) {
        _image = File(pickedFile.path);
      }

    });
    var request;
    request = http.MultipartRequest('POST',
        Uri.parse('http://192.168.74.234:8000/upload'))
      ..files.add(await http.MultipartFile.fromPath(
          'image',pickedFile!.path));
    String answersJson = jsonEncode(answers);

    // Add the JSON string as a field
    request.fields['Data'] = answersJson.toString();

    var response = await request.send();

    if (response.statusCode == 200) {
      // Handle success

      Map<String, dynamic>? dataMap = (await response.stream
          .transform(utf8.decoder)
          .transform(json.decoder)
          .first) as Map<String, dynamic>?;

      Navigator.of(context).push(MaterialPageRoute(
          builder: (context) => Details(
              imagePath: pickedFile!.path, result: dataMap, user1:widget.user,skin : dropdown,questions:answers)));
    } else {

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: Colors.greenAccent,
          content: Text('Failed to get result, try again!'),
        ),
      );
    }

  }
}
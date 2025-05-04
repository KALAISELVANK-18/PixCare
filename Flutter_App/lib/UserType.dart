import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:novalabs/Patient.dart';
import 'package:novalabs/main.dart';
import 'package:shared_preferences/shared_preferences.dart';

class UserType extends StatefulWidget {
  const UserType({super.key});

  @override
  State<UserType> createState() => _UserTypeState();
}

class _UserTypeState extends State<UserType> {
  List<String> list = <String>['Doctor / डॉक्टर', 'Patient / रोगी'];
  String dropdownValue = 'Doctor';
  void saveUserType() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    await prefs.setString('userType', 'Doctor');
  }

  void PatientUserType() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    await prefs.setString('userType', 'Patient');
  }

  Future<String?> getUserType() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    return prefs.getString('userType');
  }

  String? language;

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
      body: Center(
        child: Padding(
          padding: const EdgeInsets.only(top: 300),
          child: Container(
            child: Column(
              children: [
                Text(
                  language == 'hindi' ? 'आप हैं!' : 'You are!',
                  style: GoogleFonts.poppins(fontSize: 30),
                ),
                Padding(
                  padding: const EdgeInsets.all(10.0),
                  child: DropdownMenu<String>(
                    initialSelection: list.first,
                    width: MediaQuery.of(context).size.width * 0.70,
                    onSelected: (String? value) {
                      // This is called when the user selects an item.
                      setState(() {
                        dropdownValue = value!;
                      });
                    },
                    dropdownMenuEntries:
                    list.map<DropdownMenuEntry<String>>((String value) {
                      return DropdownMenuEntry<String>(
                          value: value, label: value);
                    }).toList(),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.only(top: 20),
                  child: Container(
                      width: MediaQuery.of(context).size.width * 0.5,
                      child: ElevatedButton(
                        onPressed: ()async {
                          if (dropdownValue == "Patient / रोगी") {
                            PatientUserType();
                            Navigator.of(context).push(
                              MaterialPageRoute(
                                builder: (context) => Patient(),
                              ),
                            );
                          } else {

                            saveUserType();
                            await FirebaseFirestore.instance
                                .collection('Doctor')
                                .doc(FirebaseAuth.instance.currentUser!.email)
                                .set({
                              "Head(सिर)": 0,
                              "Body(शरीर)": 0,
                              "Left Hand(बाएं हाथ)": 0,
                              "Right Hand(दाएं हाथ)": 0,
                              "Left Leg(बाएं पैर)": 0,
                              "Right Leg(दाएं पैर)": 0,
                              "userid":FirebaseAuth.instance.currentUser!.email
                            });
                            Navigator.of(context).push(
                              MaterialPageRoute(
                                builder: (context) => HomePage(),
                              ),
                            );
                          }
                        },
                        child: Text(
                          "Submit",
                          style: GoogleFonts.poppins(fontSize: 20),
                        ),
                      )),
                )
              ],
            ),
          ),
        ),
      ),
    );
  }
}
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:novalabs/Signin.dart';

class Signup extends StatefulWidget {
  const Signup({Key? key}) : super(key: key);

  @override
  State<Signup> createState() => _SignupState();
}

class _SignupState extends State<Signup> {
  final _formkey = GlobalKey<FormState>();

  @override
  Widget build(BuildContext context) {
    TextEditingController email = new TextEditingController();
    TextEditingController pass = new TextEditingController();
    TextEditingController conpass = new TextEditingController();
    TextEditingController fname = new TextEditingController();
    TextEditingController lname = new TextEditingController();

    return WillPopScope(
      onWillPop: () async => false,
      child: Scaffold(
        backgroundColor: Color.fromARGB(255, 255, 255, 255),
        body: SingleChildScrollView(
          child: Form(
            key: _formkey,
            child: Column(children: [
              SizedBox(
                height: 15,
              ),
              Column(
                crossAxisAlignment: CrossAxisAlignment.center,
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Container(
                      height: 200,
                      width: 200,
                      child: Image.asset('images/img.png')),
                  Padding(
                    padding: const EdgeInsets.only(top: 0),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.center,
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text(
                          "N",
                          style: GoogleFonts.aBeeZee(
                              textStyle: TextStyle(
                                  fontWeight: FontWeight.bold, fontSize: 35)),
                        ),
                        Image.asset(
                          "images/nova.png",
                          height: 35,
                          width: 35,
                        ),
                        Text("va",
                            style: GoogleFonts.kanit(
                              textStyle: TextStyle(fontSize: 30),
                            )),
                        Padding(
                          padding: const EdgeInsets.only(top: 8.0),
                          child: Text(
                            "l",
                            style: GoogleFonts.sacramento(
                              textStyle: TextStyle(
                                  fontSize: 32,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.amber),
                            ),
                          ),
                        ),
                        Padding(
                          padding: const EdgeInsets.only(top: 8.0),
                          child: Text(
                            "a",
                            style: GoogleFonts.sacramento(
                              textStyle: TextStyle(
                                  fontSize: 32,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.redAccent),
                            ),
                          ),
                        ),
                        Padding(
                          padding: const EdgeInsets.only(top: 8.0),
                          child: Text(
                            "bs",
                            style: GoogleFonts.sacramento(
                              textStyle: TextStyle(
                                  fontSize: 32,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.blue),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              Padding(
                padding: const EdgeInsets.only(top: 20),
                child: Text(
                  'personal skin care!',
                  style: GoogleFonts.poppins(color: Colors.black, fontSize: 23),
                  textAlign: TextAlign.center,
                ),
              ),

              Padding(
                  padding: EdgeInsets.only(top: 20, left: 20, right: 20),
                  child: TextFormField(
                    autovalidateMode: AutovalidateMode.onUserInteraction,
                    validator: (value) {
                      if (value!.isEmpty) {
                        return "Please enter email";
                      }
                      return null;
                    },
                    controller: email,
                    decoration: InputDecoration(
                        filled: true, //<-- SEE HERE
                        fillColor: Colors.white,
                        border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(10)),
                        hintText: 'Enter Email'),
                  )),
              Padding(
                  padding: EdgeInsets.only(top: 20, left: 20, right: 20),
                  child: TextFormField(
                    autovalidateMode: AutovalidateMode.onUserInteraction,
                    validator: (value) {
                      if (value!.isEmpty) {
                        return "Please enter password";
                      }
                      return null;
                    },
                    controller: pass,
                    decoration: InputDecoration(
                      filled: true, //<-- SEE HERE
                      fillColor: Colors.white,
                      border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(10)),
                      hintText: 'Enter Password',
                    ),
                    obscureText: true,
                  )),

              Padding(
                  padding: EdgeInsets.only(
                    top: 20,
                  ),
                  child: ElevatedButton(
                    style: ButtonStyle(
                      shape: MaterialStateProperty.all(RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10))),
                      padding: MaterialStateProperty.all(EdgeInsets.only(
                          top: 15, bottom: 15, right: 130, left: 130)),
                      backgroundColor: MaterialStateProperty.all(Colors.white),
                    ),
                    child: Text(
                      'Sign Up',
                      style:
                          TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                    ),
                    onPressed: () async {
                      try {
                        await FirebaseAuth.instance
                            .createUserWithEmailAndPassword(
                          email: email.text.toString(),
                          password: pass.text.toString(),
                        );

                        await FirebaseFirestore.instance
                            .collection(email.text)
                            .doc("personinfo")
                            .set({
                          "first_name": fname.text.toString(),
                          "second_name": lname.text.toString(),
                        });
                        Navigator.of(context).push(
                          MaterialPageRoute(
                            builder: (context) => const Signin(),
                          ),
                        );
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            backgroundColor: Colors.greenAccent,
                            content: Text('Login successfully!'),
                          ),
                        );
                      } on FirebaseAuthException {
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            backgroundColor: Colors.greenAccent,
                            content: Text('Please verify the credentials!'),
                          ),
                        );
                      }
                    },
                  )),
              Padding(
                padding: EdgeInsets.only(top: 20),
                child: Text(
                  'Or continue with',
                  style: GoogleFonts.poppins(fontSize: 15, color: Colors.white),
                ),
              ),

              // Row(
              //   mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              //   children: [
              //     Container(
              //       height: 20,
              //       width: 20,
              //       child: Padding(
              //         padding: EdgeInsets.only(top: 20),
              //         child: Image.asset('images/gg.png'),
              //       ),
              //     ),
              //     Container(
              //       height: 20,
              //       width: 20,
              //       child: Padding(
              //         padding: EdgeInsets.only(top: 20),
              //         child: Image.asset('images/apple.png'),
              //       ),
              //     ),
              //
              //   ],
              // ),
              SizedBox(
                height: 0,
              ),

              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Center(
                    child: Padding(
                      padding: EdgeInsets.only(top: 2),
                      child: Row(
                        children: [
                          Text('Already a user',
                              style: GoogleFonts.poppins(
                                  fontSize: 15, color: Colors.black)),
                          TextButton(
                            onPressed: () {
                              Navigator.of(context).push(
                                MaterialPageRoute(
                                  builder: (context) => const Signin(),
                                ),
                              );
                            },
                            child: Text(
                              'Login here',
                              style: GoogleFonts.poppins(
                                  fontSize: 17,
                                  color: Colors.deepPurpleAccent,
                                  fontWeight: FontWeight.bold),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ]),
          ),
        ),
      ),
    );
  }
}
